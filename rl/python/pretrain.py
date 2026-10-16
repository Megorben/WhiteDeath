"""Behavior cloning: teach the policy to imitate the scripted bot before reinforcement learning.

This is how AlphaStar started too: supervised learning on demonstrations first, then RL to improve on them.
Starting PPO from scratch, the agent has to stumble on "drill + full conveyor chain to the core" by chance;
starting from the bot's behavior, it only has to find improvements.

    python pretrain.py --episodes 200 --epochs 6 --out runs/bc.zip
    python train.py --init runs/bc.zip
"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from heuristic import HeuristicAgent
from mindustry_env import MindustryMiningEnv
from policy import MapPolicy
from train import GAMMA, ppo_kwargs


def collect(args):
    """Play episodes with the scripted bot, return observations, masks, actions and discounted returns."""
    seeds, shaping = args
    env = MindustryMiningEnv(shaping=shaping, gamma=GAMMA)
    bot = HeuristicAgent(env)
    obs_list, masks, actions, returns, delivered = [], [], [], [], []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        bot.reset()
        rewards, done = [], False
        while not done:
            action = bot.act(info)
            obs_list.append(obs.astype(np.float16))
            masks.append(env.action_masks())
            actions.append(action)
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
    return np.stack(obs_list), np.stack(masks), np.array(actions), np.array(returns, np.float32), delivered


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shaping", type=float, default=1.0, help="must match the value used in train.py")
    p.add_argument("--out", default="runs/bc.zip")
    args = p.parse_args()

    start = time.time()
    chunks = [(list(range(i, args.episodes, args.workers)), args.shaping) for i in range(args.workers)]
    with mp.Pool(args.workers) as pool:
        parts = pool.map(collect, chunks)
    obs, masks, actions, returns = (np.concatenate([part[i] for part in parts]) for i in range(4))
    delivered = [d for part in parts for d in part[4]]
    print(f"Collected {len(actions)} steps from {args.episodes} bot episodes in {time.time() - start:.0f}s, "
          f"bot delivered {np.mean(delivered):.1f} per episode", flush=True)

    env = DummyVecEnv([lambda: MindustryMiningEnv(shaping=args.shaping, gamma=GAMMA)])
    model = MaskablePPO(MapPolicy, env, device="cpu", **ppo_kwargs())
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    batch = 256
    n = len(actions)

    for epoch in range(args.epochs):
        order = np.random.permutation(n)
        total_ce = total_v = correct = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            o = torch.tensor(obs[idx], dtype=torch.float32)
            a = torch.tensor(actions[idx])
            values, log_prob, _ = policy.evaluate_actions(o, a, action_masks=masks[idx])
            ce = -log_prob.mean()
            v_loss = torch.nn.functional.mse_loss(values.flatten(), torch.tensor(returns[idx]))
            loss = ce + 0.5 * v_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                dist = policy.get_distribution(o, action_masks=masks[idx])
                correct += (dist.distribution.probs.argmax(1) == a).sum().item()
            total_ce += ce.item() * len(idx)
            total_v += v_loss.item() * len(idx)
        print(f"epoch {epoch + 1}: imitation loss {total_ce / n:.3f} | value loss {total_v / n:.3f} | "
              f"matches bot {correct / n:.1%}", flush=True)

    model.save(args.out)
    env.close()
    print("Saved", args.out)


if __name__ == "__main__":
    main()
