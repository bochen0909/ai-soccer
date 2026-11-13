"""Vectorized client for the Node environment server (scripts/env-server.mjs)."""
import json
import os
import struct
import subprocess
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INIT, RESET, STEP, CLOSE = 1, 2, 3, 4


@dataclass
class Batch:
    obs: np.ndarray         # [envs, agents, obs_size] float32
    mask: np.ndarray        # [envs, agents, act_count] bool, legal acts
    agent_mask: np.ndarray  # [envs, agents] bool, agents that exist
    reward: np.ndarray      # [envs] float32, team reward of the step
    done: np.ndarray        # [envs] bool, the episode ended (obs already belong to the next one)
    score: np.ndarray       # [envs, 2] int32, (own, opponent) goals of the ended episode
    steps: np.ndarray       # [envs] int32, length of the ended episode, or -1 if the engine failed and it was dropped


class _Server:
    def __init__(self, options):
        self.proc = subprocess.Popen(
            ["node", os.path.join(ROOT, "scripts", "env-server.mjs")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=ROOT, bufsize=0)
        self.options = options

    def send(self, kind, body=b""):
        self.proc.stdin.write(struct.pack("<IB", len(body) + 1, kind) + body)

    def read(self):
        header = self._exact(5)
        length, kind = struct.unpack("<IB", header)
        return kind, self._exact(length - 1)

    def _exact(self, count):
        data = bytearray()
        while len(data) < count:
            chunk = self.proc.stdout.read(count - len(data))
            if not chunk:
                raise RuntimeError("environment server exited")
            data.extend(chunk)
        return bytes(data)


class SoccerVecEnv:
    """Runs `processes * envs_per_process` matches. Every agent of the learner team is one row.

    Options (all optional): size (5 or 11), opponent ("default" or "tuned"), ticksPerStep, maxSteps,
    periodTicks (shorter halves), reward ({goalScored, goalConceded, ballProgress, possession}).
    """

    def __init__(self, processes=4, envs_per_process=4, seed=0, **options):
        self.servers = []
        self.layout = None
        for index in range(processes):
            server = _Server(options)
            server.send(INIT, json.dumps({**options, "envs": envs_per_process,
                                          "seedBase": seed + index * 1_000_003}).encode())
            self.servers.append(server)
        for server in self.servers:
            kind, body = server.read()
            assert kind == 129
            layout = json.loads(body)
            self.layout = self.layout or layout
        self.envs_per_process = envs_per_process
        self.num_envs = processes * envs_per_process
        self.agents = self.layout["agents"]
        self.obs_size = self.layout["obsSize"]
        self.act_count = self.layout["actCount"]
        self.move_count = self.layout["moveCount"]

    def _collect(self):
        parts = []
        for server in self.servers:
            kind, body = server.read()
            assert kind in (130, 131), kind
            parts.append(self._parse(body))
        return Batch(*[np.concatenate([getattr(p, f) for p in parts]) for f in Batch.__dataclass_fields__])

    def _parse(self, body):
        e, a, d, c = self.envs_per_process, self.agents, self.obs_size, self.act_count
        pad = lambda n: (n + 3) & ~3
        offset = 0

        def take(dtype, count, nbytes):
            nonlocal offset
            array = np.frombuffer(body, dtype=dtype, count=count, offset=offset)
            offset += nbytes
            return array

        obs = take(np.float32, e * a * d, e * a * d * 4).reshape(e, a, d)
        mask = take(np.uint8, e * a * c, pad(e * a * c)).reshape(e, a, c).astype(bool)
        agent_mask = take(np.uint8, e * a, pad(e * a)).reshape(e, a).astype(bool)
        reward = take(np.float32, e, e * 4)
        done = take(np.uint8, e, pad(e)).astype(bool)
        score = take(np.int32, e * 2, e * 8).reshape(e, 2)
        steps = take(np.int32, e, e * 4)
        return Batch(obs.copy(), mask, agent_mask, reward.copy(), done, score.copy(), steps.copy())

    def reset(self, seed=None):
        for index, server in enumerate(self.servers):
            options = {} if seed is None else {"seedBase": seed + index * 1_000_003, "episode": 0}
            server.send(RESET, json.dumps(options).encode())
        return self._collect()

    def step(self, actions):
        """actions: int array [envs, agents, 3] holding (move, sprint, act) per agent."""
        actions = np.ascontiguousarray(actions, dtype="<i4").reshape(self.num_envs, self.agents, 3)
        per = self.envs_per_process
        for index, server in enumerate(self.servers):
            server.send(STEP, actions[index * per:(index + 1) * per].tobytes())
        return self._collect()

    def close(self):
        for server in self.servers:
            try:
                server.send(CLOSE)
                server.proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        for server in self.servers:
            server.proc.wait(timeout=10)
