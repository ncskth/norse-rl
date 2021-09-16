import gym
from gym import spaces
import numpy as np
import math


class GridworldEnv(gym.Env):
    """
    Description:
        ...
    Source:
        ...
    Observation:
        Type: Box(2)
        Num     Observation               Min                     Max
        0       Distance to food          0                       142
        1       Relative angle to food    -3.14                   3.14
    Actions:
        Type: MultiDiscrete([2, 2])
        Num     Action
        [0, 0]  Stand still
        [1, 0]  Rotate left
        [0, 1]  Rotate right
        [1, 1]  Move forward
    Reward:
        Reward is 0 for every step taken, 1 when standing on food source
    Starting State:
        Center, pointing north
    Episode Termination:
        None

    """

    action_space = spaces.MultiDiscrete([2, 2])  # Walk left or walk right
    observation_space = spaces.MultiDiscrete([100, 100])
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 50}
    pixel_scale = 5

    def __init__(self):
        pass

    def _draw_square(self, img, x, y, color):
        x_scaled = x * self.pixel_scale
        y_scaled = y * self.pixel_scale
        img[
            y_scaled : y_scaled + self.pixel_scale,
            x_scaled : x_scaled + self.pixel_scale,
        ] = color
        return img

    def render(self, mode="rgb_array"):
        img = np.zeros(([x.n * self.pixel_scale for x in self.observation_space]))
        self._draw_square(img, *self.state, 1)
        return img

    def reset(self):
        # Init in center pointing north
        self.state = np.array(
            [int(x.n / 2) for x in self.observation_space] + [math.pi / 2]
        )
        return self.state

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        left_move, right_move = action
        angle = self.state[-1]

        if left_move and not right_move:
            move = np.array([math.cos(angle), math.sin(angle)])
            angle -= math.pi / 2
            move += np.array([math.cos(angle), math.sin(angle)])
        elif right_move and not left_move:
            move = np.array([math.cos(angle), math.sin(angle)])
            angle += math.pi / 2
            move += np.array([math.cos(angle), math.sin(angle)])
        elif left_move and right_move:
            move = np.array([math.cos(angle), math.sin(angle)])
        else:
            move = np.array([0, 0])

        self.state = np.array([*(self.state[:2] + move), angle])

        return self.state, 0, False, {}


if __name__ == "__main__":
    w = GridworldEnv()
    w.reset()
    w.step([1, 0])
    print(w)
