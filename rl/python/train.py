"""Train a MaskablePPO agent on the Mindustry mining task.

    python pretrain.py --out runs/bc.zip           # imitate the scripted bot first (recommended)
    python train.py --init runs/bc.zip --timesteps 2000000 --envs 4
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


GAMMA = 0.99


def ppo_kwargs(learning_rate=3e-4, ent_coef=0.01):
    return dict(n_steps=512, batch_size=256, n_epochs=3, learning_rate=learning_rate, gamma=GAMMA,
                gae_lambda=0.95, ent_coef=ent_coef, clip_range=0.2)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--out", default="runs/mining")
    p.add_argument("--resume", help="continue training a saved model .zip with its own settings")
    p.add_argument("--init", help="start from the weights of a saved model, e.g. the behavior cloning one")
    p.add_argument("--lr", type=float, help="learning rate (default 3e-4, or 1e-4 with --init)")
    p.add_argument("--ent-coef", type=float, help="entropy bonus (default 0.01, or 0.001 with --init)")
    p.add_argument("--device", default="auto")
    p.add_argument("--shaping", type=float, default=1.0, help="weight of the connection shaping reward, 0 = off")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = SubprocVecEnv([make_env(i, args.out, args.shaping) for i in range(args.envs)])

    if args.resume:
        model = MaskablePPO.load(args.resume, env=env, device=args.device)
    else:
        # fine-tuning a pretrained policy needs smaller steps and less exploration, or PPO wrecks it early on
        lr = args.lr or (1e-4 if args.init else 3e-4)
        ent_coef = args.ent_coef if args.ent_coef is not None else (0.001 if args.init else 0.01)
        model = MaskablePPO(MapPolicy, env, tensorboard_log=args.out, device=args.device, verbose=0,
                            **ppo_kwargs(lr, ent_coef))
        if args.init:
            model.policy.load_state_dict(MaskablePPO.load(args.init, device=args.device).policy.state_dict())

    callbacks = [ProgressCallback(), CheckpointCallback(100_000 // args.envs, os.path.join(args.out, "checkpoints"), "mining")]
    try:
        model.learn(args.timesteps, callback=callbacks, reset_num_timesteps=not args.resume)
    finally:
        model.save(os.path.join(args.out, "final"))
        env.close()
        print("Saved", os.path.join(args.out, "final.zip"))


if __name__ == "__main__":
    main()
