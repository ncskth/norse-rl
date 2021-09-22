import math
from random import random
import numpy as np

from norse_rl.env import GridworldEnv

MIDX = 250
MIDY = 250


def test_step_stay():
    w = GridworldEnv(dt=0.5)
    w.reset()
    w.step([0, 0], random_movement=False)
    assert np.allclose(w.state, np.array([MIDX, MIDY, 0]))


def test_step_left():
    w = GridworldEnv(dt=0.5)
    w.reset()
    w.step([1, 0], random_movement=False)
    assert np.allclose(w.state, np.array([MIDX, MIDY, -w.dt * w.ROTATION_SCALE]))
    w.step([1, 0], random_movement=False)
    assert np.allclose(w.state, np.array([MIDX, MIDY, -w.dt * w.ROTATION_SCALE * 2]))


def test_step_right():
    w = GridworldEnv(dt=0.5)
    w.reset()
    w.step([0, 1], random_movement=False)
    assert np.allclose(w.state, np.array([MIDX, MIDY, w.dt * w.ROTATION_SCALE]))
    w.step([0, 1], random_movement=False)
    assert np.allclose(w.state, np.array([MIDX, MIDY, w.dt * w.ROTATION_SCALE * 2]))


def test_step_forward():
    w = GridworldEnv(dt=0.5)
    w.reset()
    w.step([1, 1], random_movement=False)
    assert np.allclose(w.state, np.array([MIDX + 0.5, MIDY, 0]))


def test_step_forward_rotated():
    w = GridworldEnv(dt=0.5)
    w.reset()
    w.state = np.array([*w.state[:2], -math.pi])  # W
    w.step([1, 1], random_movement=False)
    assert np.allclose(w.state, np.array([MIDX - 0.5, MIDY, math.pi]))


def test_food_pos():
    w = GridworldEnv(dt=0.5)
    w.reset()
    w.food = [(MIDX, MIDY - 1)]
    assert w._closest_food((MIDX, MIDY)) == (w.DIST_SCALE, (MIDX, MIDY - 1))


def test_food_pos_diagonal():
    w = GridworldEnv(dt=0.5)
    w.reset()
    w.food = [(31, 30), (0, 0)]
    assert w._closest_food((32, 32)) == (2.23606797749979 * w.DIST_SCALE, (31, 30))


def test_step_observation():
    w = GridworldEnv(dt=0.5)
    w.reset()
    w.food = [(MIDX, MIDY - 10)]
    obs, rew, end, _ = w.step([0, 0], random_movement=False)
    assert np.allclose(obs, np.array([math.pi / 2 * w.ROTATION_SCALE, 0]), atol=1e-4)
    assert rew == 0
    assert not end


def test_step_observation_relative_angle_left():
    w = GridworldEnv(dt=0.5)
    w.reset()
    w.food = [(MIDX, MIDY - 10)]
    w.state = np.array([*w.state[:2], -math.pi / 4])  # SE
    obs, rew, end, _ = w.step([0, 0], random_movement=False)
    assert np.allclose(
        obs, np.array([math.pi * (3 / 4) * w.ROTATION_SCALE, 0]), atol=1e-4
    )


def test_step_observation_relative_angle_right():
    w = GridworldEnv(dt=0.5)
    w.reset()
    w.food = [(MIDX, MIDY - 10)]
    w.state = np.array([*w.state[:2], -math.pi])  # W
    obs, rew, end, _ = w.step([0, 0], random_movement=False)
    assert np.allclose(obs, np.array([0, math.pi / 2 * w.ROTATION_SCALE]), atol=1e-4)


def test_step_observation_relative_angle_overflow():
    w = GridworldEnv(dt=0.5)
    w.reset()
    w.food = [(MIDX, MIDY - 10)]
    w.state = np.array([*w.state[:2], math.pi * (3 / 4)])  # NW
    obs, rew, end, _ = w.step([0, 0], random_movement=False)
    assert np.allclose(obs, np.array([0, math.pi / 4 * w.ROTATION_SCALE]), atol=1e-4)


def test_step_observation_relative_angle_overflow_opposite():
    w = GridworldEnv(dt=0.5)
    w.reset()
    w.food = [(420, 380)]
    w.state = np.array([400, 400, -math.pi * (3 / 4)])  # SW
    obs, rew, end, _ = w.step([0, 0], random_movement=False)
    assert np.allclose(obs, np.array([math.pi * w.ROTATION_SCALE, 0]), atol=1e-4)
