"""Gymnasium environment backed by the headless Mindustry simulator (rl/sim)."""
import base64
import json
import os
import subprocess

import gymnasium as gym
import numpy as np
from gymnasium import spaces

SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim")
SERVER_JAR = os.path.join(SIM_DIR, "lib", "server-v146.jar")

# Must match SimServer.java
CH_ORE_COPPER, CH_ORE_LEAD, CH_WALL, CH_CORE, CH_DRILL, CH_CONVEYOR, CH_HAS_ITEMS = 0, 1, 2, 3, 4, 5, 9
SIM_CHANNELS = 10
EXTRA_CHANNELS = 3  # copper, lead, episode progress as constant planes
NOOP = 0


class MindustryMiningEnv(gym.Env):
    """Place mechanical drills and conveyors to deliver copper and lead to the core.

    Actions (Discrete): 0 = do nothing, 1 + cell = drill at cell, 1 + S*S + cell*4 + rot = conveyor.
    Cell index is y * size + x; rotation 0 = right, 1 = up, 2 = left, 3 = down.
    Use action_masks() (sb3-contrib MaskablePPO) to only pick valid, affordable placements.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size=24, ticks_per_step=60, max_steps=150, start_copper=120,
                 java="java", log_path=None, render_mode=None):
        self.size = size
        self.args = ["--size", size, "--ticks", ticks_per_step, "--steps", max_steps, "--copper", start_copper]
        self.java = java
        self.log_path = log_path
        self.render_mode = render_mode
        self.n_actions = 1 + size * size * 5
        self.observation_space = spaces.Box(0.0, 1.0, (SIM_CHANNELS + EXTRA_CHANNELS, size, size), np.float32)
        self.action_space = spaces.Discrete(self.n_actions)
        self._proc = None
        self._last = None

    # ---- process management ----

    def _start(self):
        if not os.path.exists(os.path.join(SIM_DIR, "build", "whitedeath", "rl", "SimServer.class")):
            raise RuntimeError("Simulator is not built. Run rl/sim/build.sh first.")
        cmd = [self.java, "-cp", os.pathsep.join([os.path.join(SIM_DIR, "build"), SERVER_JAR]),
               "whitedeath.rl.SimServer"] + [str(a) for a in self.args]
        stderr = open(self.log_path, "a") if self.log_path else subprocess.DEVNULL
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                      text=True, bufsize=1)

    def _send(self, command):
        if self._proc is None or self._proc.poll() is not None:
            self._start()
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Simulator exited unexpectedly (pass log_path=... to see its log)")
        data = json.loads(line)
        if "error" in data:
            raise RuntimeError("Simulator error: " + data["error"])
        self._last = data
        return data

    # ---- gym API ----

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        data = self._send(f"reset {int(self.np_random.integers(0, 2**62))}")
        return self._obs(data), self._info(data)

    def step(self, action):
        data = self._send(f"step {int(action)}")
        return self._obs(data), float(data["reward"]), False, bool(data["done"]), self._info(data)

    def action_masks(self):
        return np.frombuffer(base64.b64decode(self._last["mask"]), dtype=np.uint8).astype(bool)

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write("close\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()
        self._proc = None

    # ---- helpers ----

    def grid(self):
        """Raw simulator grid of the last observation, shape (channels, size, size), indexed [c, y, x]."""
        raw = np.frombuffer(base64.b64decode(self._last["obs"]), dtype=np.uint8)
        return raw.reshape(SIM_CHANNELS, self.size, self.size)

    def _obs(self, data):
        s = self.size
        extra = np.empty((EXTRA_CHANNELS, s, s), dtype=np.float32)
        extra[0] = min(data["copper"] / 500.0, 1.0)
        extra[1] = min(data["lead"] / 500.0, 1.0)
        extra[2] = data["step"] / data["max_steps"]
        return np.concatenate([self.grid().astype(np.float32), extra])

    @staticmethod
    def _info(data):
        return {k: data[k] for k in ("copper", "lead", "income", "total_income", "step")}

    def render(self):
        g = self.grid()
        chars = []
        for y in reversed(range(self.size)):
            row = ""
            for x in range(self.size):
                if g[CH_CORE, y, x]: c = "C"
                elif g[CH_DRILL, y, x]: c = "D"
                elif any(g[CH_CONVEYOR + r, y, x] for r in range(4)):
                    c = ">^<v"[next(r for r in range(4) if g[CH_CONVEYOR + r, y, x])]
                elif g[CH_WALL, y, x]: c = "#"
                elif g[CH_ORE_COPPER, y, x]: c = "c"
                elif g[CH_ORE_LEAD, y, x]: c = "l"
                else: c = "."
                row += c
            chars.append(row)
        d = self._last
        chars.append(f"step {d['step']}/{d['max_steps']}  copper {d['copper']}  lead {d['lead']}  delivered {d['total_income']}")
        return "\n".join(chars)


def drill_action(size, x, y):
    return 1 + y * size + x


def conveyor_action(size, x, y, rotation):
    return 1 + size * size + (y * size + x) * 4 + rotation
