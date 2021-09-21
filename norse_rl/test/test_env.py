import math
import numpy as np

from norse_rl.env import GridworldEnv

MIDX = 250
MIDY = 250

def test_step_stay():
   w = GridworldEnv(dt = 0.5)
   w.reset()
   w.step([0, 0])
   assert np.allclose(w.state, np.array([MIDX, MIDY, 0]))

def test_step_left():
   w = GridworldEnv(dt = 0.5)
   w.reset()
   w.step([1, 0])
   assert np.allclose(w.state, np.array([MIDX, MIDY, - w.dt / math.pi * 2]))

def test_step_right():
   w = GridworldEnv(dt = 0.5)
   w.reset()
   w.step([0, 1])
   assert np.allclose(w.state, np.array([MIDX, MIDY, w.dt / math.pi * 2]))

def test_step_forward():
   w = GridworldEnv(dt = 0.5)
   w.reset()
   w.step([1, 1])
   assert np.allclose(w.state, np.array([MIDX + 0.5, MIDY, 0]))

def test_food_pos():
   w = GridworldEnv(dt = 0.5)
   w.reset()
   w.food = [(MIDX, MIDY - 1)]
   assert w._closest_food((MIDX, MIDY)) == (w.DIST_SCALE, (MIDX, MIDY - 1))

def test_food_pos_diagonal():
   w = GridworldEnv(dt = 0.5)
   w.reset()
   w.food = [(31, 30), (0, 0)]
   assert w._closest_food((32, 32)) == (2.23606797749979 * w.DIST_SCALE, (31, 30))

def test_step_observation():
   w = GridworldEnv(dt = 0.5)
   w.reset()
   w.food = [(MIDX, MIDY - 10)]
   obs, rew, end, _ = w.step([0, 0])
   assert np.allclose(obs, np.array([10 * w.DIST_SCALE, math.pi / 2]), atol=1e-4)
   assert rew == 0
   assert not end