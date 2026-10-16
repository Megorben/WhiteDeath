"""Scripted baseline: put a drill on the richest reachable ore and route a conveyor to the core with BFS.

The RL agent has to beat this bot to be worth anything.
"""
from collections import deque

import numpy as np

from mindustry_env import (CH_CONVEYOR, CH_CORE, CH_DRILL, CH_ORE_COPPER, CH_ORE_LEAD, CH_WALL, NOOP,
                           conveyor_action, drill_action)

DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # index == conveyor rotation
DRILL_COST, CONVEYOR_COST = 12, 1


class HeuristicAgent:
    def __init__(self, env):
        self.env = env
        self.size = env.size
        self.plan = deque()

    def reset(self):
        self.plan.clear()

    def act(self, info):
        mask = self.env.action_masks()
        while self.plan:
            if mask[self.plan[0]]:
                return self.plan.popleft()
            if not self._affordable(self.plan[0], info):
                return NOOP  # wait for resources
            self.plan.clear()  # plan became invalid, replan
        self._make_plan(mask, info)
        return self.plan.popleft() if self.plan and mask[self.plan[0]] else NOOP

    def _affordable(self, action, info):
        return info["copper"] >= (DRILL_COST if action <= self.size * self.size else CONVEYOR_COST)

    def _make_plan(self, mask, info):
        g = self.env.grid()
        s = self.size
        ore = g[CH_ORE_COPPER] + g[CH_ORE_LEAD]
        occupied = (g[CH_WALL] + g[CH_CORE] + g[CH_DRILL] + g[CH_CONVEYOR:CH_CONVEYOR + 4].sum(0)) > 0
        core = set(zip(*np.nonzero(g[CH_CORE])[::-1]))  # (x, y)

        best = None
        for y in range(s):
            for x in range(s):
                if not mask[drill_action(s, x, y)]:
                    continue
                footprint = {(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)}
                ores = int(sum(ore[fy, fx] for fx, fy in footprint))
                path = self._route(footprint, core, occupied)
                if path is None:
                    continue
                cost = DRILL_COST + len(path) * CONVEYOR_COST
                score = ores * 10 - len(path)
                if cost <= info["copper"] and (best is None or score > best[0]):
                    best = (score, x, y, path)

        if best is None:
            return
        _, x, y, path = best
        self.plan.append(drill_action(s, x, y))
        for (px, py), rot in path:
            self.plan.append(conveyor_action(s, px, py, rot))

    def _route(self, footprint, core, occupied):
        """Shortest conveyor path from the drill to the core. Returns [((x, y), rotation)], [] if adjacent."""
        s = self.size
        if any((fx + dx, fy + dy) in core for fx, fy in footprint for dx, dy in DIRS):
            return []
        start = [(fx + dx, fy + dy) for fx, fy in footprint for dx, dy in DIRS]
        start = [p for p in start if p not in footprint and 0 <= p[0] < s and 0 <= p[1] < s and not occupied[p[1], p[0]]]
        prev = {p: None for p in start}
        queue = deque(start)
        while queue:
            p = queue.popleft()
            for rot, (dx, dy) in enumerate(DIRS):
                n = (p[0] + dx, p[1] + dy)
                if n in core:
                    path = [(p, rot)]
                    while prev[p] is not None:
                        q = prev[p]
                        path.append((q, DIRS.index((p[0] - q[0], p[1] - q[1]))))
                        p = q
                    return path[::-1]
                if (0 <= n[0] < s and 0 <= n[1] < s and n not in prev and n not in footprint
                        and not occupied[n[1], n[0]]):
                    prev[n] = p
                    queue.append(n)
        return None
