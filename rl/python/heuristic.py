"""Scripted baseline bot. The RL agent has to beat it to be worth anything.

The logic lives in the simulator (SimServer.expertAction) because it needs exact building data:
lay the next conveyor of the unconnected drill closest to the core (BFS over free cells, merging into
existing chains), otherwise place a drill on the richest ore that is cheap to connect, otherwise wait.
It decides from the current state only, so it can also label states the learning agent visits (DAgger).
"""


class HeuristicAgent:
    def __init__(self, env):
        self.env = env

    def reset(self):
        pass

    def act(self, info):
        return info["expert"]
