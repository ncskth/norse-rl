import math
import numpy as np

from rl.env import GridworldEnv

def test_step_stay():
   w = GridworldEnv()
   w.reset()
   w.step([0, 0])
   assert np.allclose(w.state, np.array([50, 50, math.pi / 2]))

def test_step_left():
   w = GridworldEnv()
   w.reset()
   w.step([1, 0])
   assert np.allclose(w.state, np.array([51, 51, 0]))

def test_step_right():
   w = GridworldEnv()
   w.reset()
   w.step([0, 1])
   assert np.allclose(w.state, np.array([49, 51, math.pi]))

def test_step_forward():
   w = GridworldEnv()
   w.reset()
   w.step([1, 1])
   assert np.allclose(w.state, np.array([50, 51, math.pi / 2]))