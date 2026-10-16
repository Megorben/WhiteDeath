"""Compare agents on the same maps: random valid actions, the scripted bot and a trained model.

    python evaluate.py --model runs/mining/final.zip --episodes 20 --render
"""
import argparse

import numpy as np

from heuristic import HeuristicAgent
from mindustry_env import MindustryMiningEnv


def run(env, choose, episodes, seed, render=False, reset=None):
    results = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        if reset:
            reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(choose(obs, info))
            done = term or trunc
        results.append(info["total_income"])
        if render and ep == 0:
            print(env.render(), "\n")
    return np.array(results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="trained MaskablePPO .zip")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=10_000, help="maps are never seen in training with this offset")
    p.add_argument("--render", action="store_true", help="print the first map of each agent")
    args = p.parse_args()

    env = MindustryMiningEnv()
    rng = np.random.default_rng(0)
    agents = {"random": (lambda o, i: rng.choice(np.flatnonzero(env.action_masks())), None)}
    bot = HeuristicAgent(env)
    agents["scripted bot"] = (lambda o, i: bot.act(i), bot.reset)
    if args.model:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(args.model, device="cpu")
        agents["RL agent"] = (lambda o, i: int(model.predict(o, action_masks=env.action_masks(), deterministic=True)[0]), None)

    for name, (choose, reset) in agents.items():
        if args.render:
            print(f"=== {name} ===")
        r = run(env, choose, args.episodes, args.seed, args.render, reset)
        print(f"{name:>13}: delivered {r.mean():6.1f} ± {r.std():5.1f}  (min {r.min()}, max {r.max()})")
    env.close()


if __name__ == "__main__":
    main()
