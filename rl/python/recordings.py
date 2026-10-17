"""Read matches recorded by the WhiteDeath Recorder client mod (recorder/).

    python recordings.py ~/.local/share/Mindustry/whitedeath-recordings          # all matches
    python recordings.py <folder> --1v1                                           # only 1v1 PvP

Each match folder holds events.jsonl.gz (one JSON event per line, "t" in game ticks, 60 per second)
and keyframe-TICK.msav world snapshots. See recorder/README.md for the event format.
"""
import argparse
import base64
import gzip
import json
import os
from collections import defaultdict

import numpy as np


def events(match_dir):
    """Yield the match's events in order. Tolerates a truncated last line (game closed mid-match)."""
    with gzip.open(os.path.join(match_dir, "events.jsonl.gz"), "rt", encoding="utf-8") as f:
        try:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        return
        except (EOFError, gzip.BadGzipFile):
            return


def terrain(start):
    """Floor, ore overlay and static wall ids from the start event, each an (h, w) uint16 array indexed [y, x]."""
    h, w = start["h"], start["w"]
    return {name: np.frombuffer(base64.b64decode(data), dtype="<u2").reshape(h, w)
            for name, data in start["terrain"].items()}


def summarize(match_dir):
    start, end = None, None
    players = defaultdict(set)  # team id -> pseudonyms seen playing for it
    built = defaultdict(int)
    max_units = defaultdict(int)
    last_t, keyframes = 0, 0
    for e in events(match_dir):
        last_t = e["t"]
        kind = e["e"]
        if kind == "start":
            start = e
        elif kind == "end":
            end = e
        elif kind == "players":
            for p in e["p"]:
                if not p["me"]:
                    players[p["team"]].add(p["p"])
        elif kind == "built" and not e["breaking"]:
            built[e["team"]] += 1
        elif kind == "units":
            count = defaultdict(int)
            for u in e["u"]:
                count[u[2]] += 1
            for team, n in count.items():
                max_units[team] = max(max_units[team], n)
        elif kind == "keyframe":
            keyframes += 1
    if start is None:
        return None
    teams = {t["id"]: t["name"] for t in start["teams"]}
    # sides = teams that own a core and have at least one player
    sides = {t: sorted(p) for t, p in players.items() if t in teams and p}
    return dict(
        folder=os.path.basename(match_dir), map=start["map"], game=start["game"],
        pvp=start["rules"]["pvp"], minutes=round(last_t / 3600, 1),
        sides={teams[t]: len(p) for t, p in sides.items()},
        is_1v1=start["rules"]["pvp"] and len(sides) == 2 and all(len(p) == 1 for p in sides.values()),
        winner=end["winner"] if end else None, ended=end["reason"] if end else "incomplete",
        built={teams.get(t, t): n for t, n in built.items()},
        max_units={teams.get(t, t): n for t, n in max_units.items()},
        keyframes=keyframes, players={teams[t]: p for t, p in sides.items()},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="whitedeath-recordings folder, or a single match folder")
    p.add_argument("--1v1", dest="only_1v1", action="store_true", help="only show 1v1 PvP matches")
    args = p.parse_args()

    folder = os.path.expanduser(args.folder)
    dirs = [folder] if os.path.exists(os.path.join(folder, "events.jsonl.gz")) else sorted(
        os.path.join(folder, d) for d in os.listdir(folder) if os.path.exists(os.path.join(folder, d, "events.jsonl.gz")))
    shown = 0
    for d in dirs:
        s = summarize(d)
        if s is None or (args.only_1v1 and not s["is_1v1"]):
            continue
        shown += 1
        sides = ", ".join(f"{team}: {n}" for team, n in s["sides"].items()) or "no players seen"
        print(f"{s['folder']}\n  map {s['map']} | {s['minutes']} min | players {sides} | 1v1: {s['is_1v1']} | "
              f"winner {s['winner']} ({s['ended']})\n  buildings placed {s['built']} | max units {s['max_units']} | "
              f"keyframes {s['keyframes']}")
    print(f"{shown} of {len(dirs)} matches shown")


if __name__ == "__main__":
    main()
