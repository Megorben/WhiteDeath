"""Train a MaskablePPO agent on the Mindustry mining task.

    python pretrain.py --out runs/bc.zip           # imitate the scripted bot first (recommended)
    python train.py --init runs/bc.zip --timesteps 2000000 --envs 4
    tensorboard --logdir runs
"""
import argparse
import os
import time

import numpy as np
import torch

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from mindustry_env import MindustryMiningEnv
from policy import MapPolicy


GAMMA = 0.99


def ppo_kwargs(learning_rate=3e-4, ent_coef=0.01, target_kl=None):
    return dict(n_steps=512, batch_size=256, n_epochs=3, learning_rate=learning_rate, gamma=GAMMA,
                gae_lambda=0.95, ent_coef=ent_coef, clip_range=0.2, target_kl=target_kl)


def make_env(rank, log_dir, shaping):
    def init():
        env = MindustryMiningEnv(shaping=shaping, gamma=GAMMA, log_path=os.path.join(log_dir, f"sim-{rank}.log"))
        return Monitor(env, info_keywords=("total_income",))
    return init


class ProgressCallback(BaseCallback):
    """Prints mean resources delivered per episode, the number that actually matters."""

    def __init__(self, every=10_000):
        super().__init__()
        self.every = every
        self.next = every
        self.start = time.time()

    def _on_step(self):
        if self.num_timesteps >= self.next and self.model.ep_info_buffer:
            self.next += self.every
            delivered = [ep["total_income"] for ep in self.model.ep_info_buffer]
            mean = sum(delivered) / len(delivered)
            self.logger.record("mining/delivered_mean", mean)
            fps = self.num_timesteps / (time.time() - self.start)
            print(f"{self.num_timesteps:>9} steps | delivered per episode: {mean:6.1f} | {fps:.0f} steps/s", flush=True)
        return True


class ExpertImitation(BaseCallback):
    """Keeps imitating the scripted expert during RL (DAPG / kickstarting style).

    After every PPO update, one pass of imitation loss on the expert's actions for the states the agent
    just visited. Plain PPO fine-tuning slowly forgot the DAgger policy (~230 -> ~175 items per episode
    in 80k steps, even KL-limited); the imitation term holds the expert's level while the RL term looks
    for improvements on top of it.
    """

    def __init__(self, coef, batch_size=256):
        super().__init__()
        self.coef = coef
        self.batch_size = batch_size
        self.labels, self.data = [], None

    def _on_training_start(self):
        # expert action for each env's current observation
        self.current = [info["expert"] for info in self.training_env.reset_infos]

    def _on_rollout_start(self):
        if self.data is not None:
            self._imitate()  # runs right after the previous PPO update
        self.labels = []

    def _on_step(self):
        # the buffer stored the observation from before this step, which self.current describes;
        # the infos already describe the next observation (or the reset one after an episode ends)
        self.labels.append(list(self.current))
        for i, info in enumerate(self.locals["infos"]):
            done = self.locals["dones"][i]
            self.current[i] = self.training_env.reset_infos[i]["expert"] if done else info["expert"]
        return True

    def _on_rollout_end(self):
        buf = self.model.rollout_buffer
        self.data = (buf.observations.reshape(-1, *buf.obs_shape),
                     buf.action_masks.reshape(-1, buf.action_masks.shape[-1]),
                     np.array(self.labels).reshape(-1))

    def _imitate(self):
        obs, masks, labels = self.data
        policy = self.model.policy
        order = np.random.permutation(len(labels))
        total = 0.0
        for i in range(0, len(labels), self.batch_size):
            idx = order[i:i + self.batch_size]
            o = torch.as_tensor(obs[idx], device=policy.device)
            a = torch.as_tensor(labels[idx], device=policy.device)
            _, log_prob, _ = policy.evaluate_actions(o, a, action_masks=masks[idx])
            loss = -self.coef * log_prob.mean()
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy.optimizer.step()
            total += loss.item() * len(idx)
        self.logger.record("imitation/loss", total / len(labels))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--out", default="runs/mining")
    p.add_argument("--resume", help="continue training a saved model .zip with its own settings")
    p.add_argument("--init", help="start from the weights of a saved model, e.g. the behavior cloning one")
    p.add_argument("--lr", type=float, help="learning rate (default 3e-4, or 3e-5 with --init)")
    p.add_argument("--ent-coef", type=float, help="entropy bonus (default 0.01, or 0.001 with --init)")
    p.add_argument("--target-kl", type=float, help="stop an update early past this KL (default none, or 0.01 with --init)")
    p.add_argument("--imitation-coef", type=float,
                   help="weight of the expert imitation loss during RL (default 0, or 1.0 with --init)")
    p.add_argument("--device", default="auto")
    p.add_argument("--shaping", type=float, default=1.0, help="weight of the connection shaping reward, 0 = off")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = SubprocVecEnv([make_env(i, args.out, args.shaping) for i in range(args.envs)])

    if args.resume:
        model = MaskablePPO.load(args.resume, env=env, device=args.device)
    else:
        # Fine-tuning a pretrained policy needs small, KL-limited steps and little exploration.
        # With lr 1e-4 and no KL limit, updates moved the policy by KL 0.02-0.04 and the DAgger agent
        # degraded from ~230 to ~170 items per episode within 40k steps.
        lr = args.lr or (3e-5 if args.init else 3e-4)
        ent_coef = args.ent_coef if args.ent_coef is not None else (0.001 if args.init else 0.01)
        target_kl = args.target_kl if args.target_kl is not None else (0.01 if args.init else None)
        model = MaskablePPO(MapPolicy, env, tensorboard_log=args.out, device=args.device, verbose=0,
                            **ppo_kwargs(lr, ent_coef, target_kl))
        if args.init:
            model.policy.load_state_dict(MaskablePPO.load(args.init, device=args.device).policy.state_dict())

    callbacks = [ProgressCallback(), CheckpointCallback(100_000 // args.envs, os.path.join(args.out, "checkpoints"), "mining")]
    imitation = args.imitation_coef if args.imitation_coef is not None else (1.0 if args.init else 0.0)
    if imitation > 0:
        callbacks.append(ExpertImitation(imitation))
    try:
        model.learn(args.timesteps, callback=callbacks, reset_num_timesteps=not args.resume)
    finally:
        model.save(os.path.join(args.out, "final"))
        env.close()
        print("Saved", os.path.join(args.out, "final.zip"))


if __name__ == "__main__":
    main()
