"""Record agent and scripted-bot episodes and build the browser replay viewer.

    python record_replays.py --model ../models/mining.zip --out ../viewer/replays.html

Open the resulting HTML file in any browser: no server or installs needed.
"""
import argparse
import json
import os

import numpy as np
from sb3_contrib import MaskablePPO

from mindustry_env import CH_CORE, CH_HAS_ITEMS, CH_ORE_COPPER, CH_ORE_LEAD, CH_WALL, MindustryMiningEnv

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "viewer", "viewer.html")


def terrain(env):
    g = env.grid()
    rows = []
    for y in range(env.size):
        row = ""
        for x in range(env.size):
            row += ("C" if g[CH_CORE, y, x] else "#" if g[CH_WALL, y, x] else
                    "c" if g[CH_ORE_COPPER, y, x] else "l" if g[CH_ORE_LEAD, y, x] else ".")
        rows.append(row)
    return rows


def play(env, seed, choose):
    obs, info = env.reset(seed=seed)
    start = terrain(env)
    rec = dict(actions=[], copper=[], lead=[], delivered=[], flow=[])
    done = False
    while not done:
        action = int(choose(obs, info))
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        rec["actions"].append(action)
        rec["copper"].append(info["copper"])
        rec["lead"].append(info["lead"])
        rec["delivered"].append(info["total_income"])
        rec["flow"].append(np.packbits(env.grid()[CH_HAS_ITEMS].flatten()).tobytes().hex())
    return start, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="../models/mining.zip")
    p.add_argument("--out", default="../viewer/replays.html")
    p.add_argument("--maps", type=int, default=20)
    p.add_argument("--seed", type=int, default=10_000, help="same unseen maps as evaluate.py")
    args = p.parse_args()

    env = MindustryMiningEnv()
    model = MaskablePPO.load(args.model, device="cpu")
    maps = []
    for seed in range(args.seed, args.seed + args.maps):
        start, agent = play(env, seed, lambda o, i: model.predict(o, action_masks=env.action_masks(), deterministic=True)[0])
        _, bot = play(env, seed, lambda o, i: i["expert"])
        maps.append(dict(seed=seed, terrain=start, agent=agent, bot=bot))
        print(f"map {seed}: agent {agent['delivered'][-1]}, bot {bot['delivered'][-1]}", flush=True)
    env.close()

    data = json.dumps(dict(size=env.size, startCopper=120, maps=maps), separators=(",", ":"))
    html = open(TEMPLATE, encoding="utf-8").read().replace("/*REPLAYS*/null", data.replace("</", "<\\/"))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
