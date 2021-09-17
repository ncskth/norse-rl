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
        0       Distance to food          0                       91
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
    observation_space = spaces.MultiDiscrete([64, 64])
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 50}
    pixel_scale = 10

    def __init__(self, food_items: int = 10, dt: float = 0.5):
        assert food_items < 64, "Food must be < 64"
        self.food_items = food_items
        self.dt = dt

    def _draw_square(self, img, x, y, color):
        x_scaled = int(x * self.pixel_scale)
        y_scaled = int(y * self.pixel_scale)
        img[
            y_scaled : y_scaled + self.pixel_scale,
            x_scaled : x_scaled + self.pixel_scale,
        ] = color
        return img

    def _distribute_food(self):
        prob = np.random.random(
            ([x.n * self.pixel_scale for x in self.observation_space])
        )
        highest_indices = np.unravel_index(np.argsort(prob, axis=None), prob.shape)
        self.food = list(zip(highest_indices[0][:self.food_items], highest_indices[1][:self.food_items]))


    def render(self, mode="rgb_array"):
        img = np.zeros((*[x.n * self.pixel_scale for x in self.observation_space], 3))
        # Draw food
        for (x, y) in self.food:
            self._draw_square(img, x / self.pixel_scale, y / self.pixel_scale, [0, 1, 0])
        # Draw agent
        self._draw_square(img, *self.state[:2], [1, 0, 0])
        return img

    def reset(self):
        # Init in center pointing north
        self.state = np.array(
            [int(x.n / 2) for x in self.observation_space] + [0]
        )
        self._distribute_food()
        return self.state

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        left_move, right_move = action
        angle = self.state[-1]

        if left_move and not right_move:
            self.state = np.array([*self.state[:2], angle + math.pi / 2 * self.dt])
        elif right_move and not left_move:
            self.state = np.array([*self.state[:2], angle - math.pi / 2 * self.dt])
        elif left_move and right_move:
            move = np.array([math.cos(angle), math.sin(angle)])
            # Ignore moves if angle is not aligned with axes
            if move.sum() > 0.5:
                self.state = np.array([*(self.state[:2] + move), angle])

        # Define observation
        observation = np.array([0, 0])

        # Define reward
        # 0: if not on food
        # 1: if on food source, delete food

        return observation, 0, False, {}


if __name__ == "__main__":
    w = GridworldEnv()
    w.reset()
    w.step([1, 0])
    print(w)
