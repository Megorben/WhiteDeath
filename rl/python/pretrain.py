"""Imitation learning from the scripted bot (DAgger), before reinforcement learning.

This is how AlphaStar started too: supervised learning on demonstrations first, then RL to improve on them.
Starting PPO from scratch, the agent has to stumble on "drill + full conveyor chain to the core" by chance;
starting from the bot's behavior, it only has to find improvements.

Plain behavior cloning only sees states the bot visits. One wrong conveyor by the student leads to states
the bot never produced, errors compound and a chain of ten conveyors almost never gets finished.
DAgger fixes that: after the first round the student plays itself and the bot labels every state it visits
with the right action, so the student also learns how to recover from its own mistakes.

    python pretrain.py --out runs/dagger.zip
    python train.py --init runs/dagger.zip
"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from mindustry_env import MindustryMiningEnv
from policy import MapPolicy
from train import GAMMA, ppo_kwargs


def collect(job):
    """Play episodes; act with the student (or the bot with probability beta), label every state with the bot."""
    seeds, shaping, model_path, beta = job
    torch.set_num_threads(1)
    student = MaskablePPO.load(model_path, device="cpu") if model_path else None
    rng = np.random.default_rng(seeds[0])
    env = MindustryMiningEnv(shaping=shaping, gamma=GAMMA)
    obs_list, masks, labels, returns, delivered = [], [], [], [], []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        rewards, done = [], False
        while not done:
            mask = env.action_masks()
            obs_list.append(obs.astype(np.float16))
            masks.append(mask)
            labels.append(info["expert"])
            if student is None or rng.random() < beta:
                action = info["expert"]
            else:
                action = int(student.predict(obs, action_masks=mask, deterministic=False)[0])
            obs, reward, term, trunc, info = env.step(action)
            rewards.append(reward)
            done = term or trunc
        ret, ep_returns = 0.0, []
        for r in reversed(rewards):
            ret = r + GAMMA * ret
            ep_returns.append(ret)
        returns.extend(reversed(ep_returns))
        delivered.append(info["total_income"])
    env.close()
    return np.stack(obs_list), np.stack(masks), np.array(labels), np.array(returns, np.float32), delivered


def gather(pool, workers, seeds, shaping, model_path, beta):
    jobs = [(seeds[i::workers], shaping, model_path, beta) for i in range(workers)]
    parts = pool.map(collect, jobs)
    data = tuple(np.concatenate([part[i] for part in parts]) for i in range(4))
    return data, [d for part in parts for d in part[4]]


def fit(policy, optimizer, data, epochs):
    obs, masks, labels, returns = data
    n, batch = len(labels), 256
    for epoch in range(epochs):
        order = np.random.permutation(n)
        total_ce = total_v = correct = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            o = torch.tensor(obs[idx], dtype=torch.float32)
            a = torch.tensor(labels[idx])
            values, log_prob, _ = policy.evaluate_actions(o, a, action_masks=masks[idx])
            ce = -log_prob.mean()
            v_loss = torch.nn.functional.mse_loss(values.flatten(), torch.tensor(returns[idx]))
            optimizer.zero_grad()
            (ce + 0.5 * v_loss).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                probs = policy.get_distribution(o, action_masks=masks[idx]).distribution.probs
                correct += (probs.argmax(1) == a).sum().item()
            total_ce += ce.item() * len(idx)
            total_v += v_loss.item() * len(idx)
        print(f"  epoch {epoch + 1}: imitation loss {total_ce / n:.3f} | value loss {total_v / n:.3f} | "
              f"matches bot {correct / n:.1%}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300, help="bot demonstrations in the first round")
    p.add_argument("--rounds", type=int, default=4, help="DAgger rounds after the first one")
    p.add_argument("--round-episodes", type=int, default=100)
    p.add_argument("--epochs", type=int, default=6, help="training epochs on the first round")
    p.add_argument("--round-epochs", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shaping", type=float, default=1.0, help="must match the value used in train.py")
    p.add_argument("--out", default="runs/dagger.zip")
    args = p.parse_args()

    env = DummyVecEnv([lambda: MindustryMiningEnv(shaping=args.shaping, gamma=GAMMA)])
    model = MaskablePPO(MapPolicy, env, device="cpu", **ppo_kwargs())
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    seed = 0

    with mp.Pool(args.workers) as pool:
        start = time.time()
        data, delivered = gather(pool, args.workers, list(range(seed, seed + args.episodes)), args.shaping, None, 1.0)
        seed += args.episodes
        print(f"Round 0: {len(data[2])} steps from {args.episodes} bot episodes in {time.time() - start:.0f}s, "
              f"bot delivered {np.mean(delivered):.1f} per episode", flush=True)
        fit(model.policy, optimizer, data, args.epochs)
        model.save(args.out)

        for r in range(1, args.rounds + 1):
            beta = 0.5 ** r  # how often the bot still drives; the student takes over gradually
            start = time.time()
            new, delivered = gather(pool, args.workers, list(range(seed, seed + args.round_episodes)),
                                    args.shaping, args.out, beta)
            seed += args.round_episodes
            data = tuple(np.concatenate([old, add]) for old, add in zip(data, new))
            print(f"Round {r}: student drove {1 - beta:.0%} of steps, delivered {np.mean(delivered):.1f} per episode; "
                  f"dataset {len(data[2])} steps ({time.time() - start:.0f}s)", flush=True)
            fit(model.policy, optimizer, data, args.round_epochs)
            model.save(args.out)

    env.close()
    print("Saved", args.out)


if __name__ == "__main__":
    main()
