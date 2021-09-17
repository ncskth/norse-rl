import math
import numpy as np

from rl.env import GridworldEnv

def test_step_stay():
   w = GridworldEnv(dt = 0.5)
   w.reset()
   w.step([0, 0])
   assert np.allclose(w.state, np.array([50, 50, math.pi / 2]))

def test_step_left():
   w = GridworldEnv(dt = 0.5)
   w.reset()
   w.step([1, 0])
   assert np.allclose(w.state, np.array([50, 50, math.pi * 3/ 4]))

def test_step_right():
   w = GridworldEnv(dt = 0.5)
   w.reset()
   w.step([0, 1])
   assert np.allclose(w.state, np.array([50, 50, math.pi / 4]))

def test_step_forward():
   w = GridworldEnv(dt = 0.5)
   w.reset()
   w.step([1, 1])
   assert np.allclose(w.state, np.array([50, 51, math.pi / 2]))