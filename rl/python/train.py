"""Train a MaskablePPO agent on the Mindustry mining task.

    python train.py --timesteps 2000000 --envs 4
    tensorboard --logdir runs
"""
import argparse
import os
import time

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from mindustry_env import MindustryMiningEnv
from policy import MapPolicy


def make_env(rank, log_dir):
    def init():
        env = MindustryMiningEnv(log_path=os.path.join(log_dir, f"sim-{rank}.log"))
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--out", default="runs/mining")
    p.add_argument("--resume", help="path to a saved model .zip")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = SubprocVecEnv([make_env(i, args.out) for i in range(args.envs)])

    if args.resume:
        model = MaskablePPO.load(args.resume, env=env, device=args.device)
    else:
        model = MaskablePPO(
            MapPolicy, env,
            n_steps=512, batch_size=256, n_epochs=4,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            ent_coef=0.01, clip_range=0.2,
            tensorboard_log=args.out, device=args.device, verbose=0,
        )

    callbacks = [ProgressCallback(), CheckpointCallback(100_000 // args.envs, os.path.join(args.out, "checkpoints"), "mining")]
    try:
        model.learn(args.timesteps, callback=callbacks, reset_num_timesteps=not args.resume)
    finally:
        model.save(os.path.join(args.out, "final"))
        env.close()
        print("Saved", os.path.join(args.out, "final.zip"))


if __name__ == "__main__":
    main()
