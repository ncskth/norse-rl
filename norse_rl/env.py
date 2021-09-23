import math
import random

import gym
from gym import spaces
import numpy as np

import norse_rl.util as util


class GridworldEnv(gym.Env):
    """
    Description:
        ...
    Source:
        ...
    Observation:
        Type: Box(2)
        Num     Observation               Min                     Max
        0       Distance to food          0                       ~2
        1       Relative angle to food    -3.14                   3.14
    Actions:
        Type: Box(2,)
        Num     Action
        0       Movement of the left feet
        1       Movement of the right feet
    Reward:
        Reward is 0 for every step taken, 1 when standing on food source
    Starting State:
        Center, pointing east
    Episode Termination:
        None
    """

    MAX_SIZE = 400
    DIST_SCALE = 1 / math.sqrt(MAX_SIZE ** 2 + MAX_SIZE ** 2)
    ROTATION_SCALE = 1 / (math.pi * 2)

    action_labels = ["Left", "Right"]
    observation_labels = ["Left Angle", "Right Angle"]

    action_space = spaces.Box(0, 1, shape=(2,))
    observation_space = spaces.MultiDiscrete([MAX_SIZE, MAX_SIZE])
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 30}
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 50}
    pixel_scale = 5

    def __init__(self, food_items: int = 10, image_scale: float = 1.0, dt: float = 1.0):
        assert food_items < self.MAX_SIZE, f"Food must be < {self.MAX_SIZE}"
        self.food_items = food_items
        self.image_scale = image_scale
        self.dt = dt

    def _draw_square(self, canvas, x, y, color, size):
        canvas.fill_style = color
        canvas.fill_rect(
            int((x - size / 2) * self.image_scale),
            int((y - size / 2) * self.image_scale),
            size,
        )
        return canvas

    def _draw_agent(self, canvas, x, y, color):
        size = 15
        self._draw_square(canvas, x, y, color, size)

        # Draw "nose"
        size = 20
        canvas.stroke_style = color
        canvas.stroke_width = 5.0
        canvas.stroke_line(
            x,
            y,
            x + math.cos(self.state[-1]) * size,
            y - math.sin(self.state[-1]) * size,
        )
        # dx = np.linspace(x, x + math.cos(self.state[-1]) * size, size)
        # dy = np.linspace(y, y - math.sin(self.state[-1]) * size, size)
        # for ax, ay in zip(dx, dy):
        #     self._draw_square(canvas, ax, ay, color, 3)
        return canvas

    def _distribute_food(self):
        prob = np.random.random(self.observation_space.nvec)
        highest_indices = np.unravel_index(np.argsort(prob, axis=None), prob.shape)
        self.food = list(
            zip(
                highest_indices[0][: self.food_items],
                highest_indices[1][: self.food_items],
            )
        )

    def _closest_food(self, position):
        min_dist = +math.inf
        min_pos = None
        for f in self.food:
            d = np.linalg.norm(f - np.array(position))
            if d < min_dist:
                min_dist = d
                min_pos = f
        return min_dist * self.DIST_SCALE, min_pos  # Scale distance [0;0.1]

    def _observe(self):
        # Define reward
        dist, food_pos = self._closest_food(self.state[:2])
        if dist < self.DIST_SCALE * 7:  # Radius of 5
            self.food.remove(food_pos)  # Delete food
            reward = 1
            dist, food_pos = self._closest_food(self.state[:2])
        else:
            reward = 0

        # Define angle to food
        x = food_pos[0] - self.state[0]
        y = food_pos[1] - self.state[1]
        target_angle = math.atan2(-y, x)
        current_angle = self.state[2]
        angle = math.atan2(
            math.sin(current_angle - target_angle),
            math.cos(current_angle - target_angle),
        )
        angle_left = max(0, -angle) * self.ROTATION_SCALE
        angle_right = max(0, angle) * self.ROTATION_SCALE

        return np.array([angle_left, angle_right]), reward

    def render(self, canvas):
        if len(self.food) == 0:
            return canvas
        # Draw food
        for (x, y) in self.food:
            self._draw_square(canvas, x, y, "rgb(246, 195, 53)", 10)
        # Draw agent
        self._draw_agent(canvas, *self.state[:2], "red")
        return canvas

    def reset(self):
        # Init in center pointing east
        self.state = np.array([x // 2 for x in self.observation_space.nvec] + [0])
        self._distribute_food()
        return self._observe()[0]

    def step(self, action, random_movement: bool = True):
        left_move, right_move = np.array(action).clip(0, 1)
        angle = self.state[-1]

        distance = (min(right_move, left_move)) * self.dt
        if random_movement:
            distance += random.uniform(0.2, 2) * self.dt
        d_rotation = (right_move - left_move) * self.dt * self.ROTATION_SCALE
        new_angle = angle + d_rotation
        if angle >= math.pi:
            new_angle = angle - 2 * math.pi
        elif angle <= -math.pi:
            new_angle = angle + 2 * math.pi

        d_x = distance * math.cos(new_angle)
        d_y = -distance * math.sin(new_angle)

        # Set new state and validate
        location = (self.state[:2] + np.array([d_x, d_y])).clip(0, self.MAX_SIZE)

        self.state = np.array([*location, new_angle])
        self.state = np.nan_to_num(self.state, 0)  # Remove NaN

        # Define observation
        is_done = len(self.food) == 0
        if is_done:
            observation, reward = np.array([0, 0]), 0
        else:
            observation, reward = self._observe()
        return observation, reward, is_done, {}


if __name__ == "__main__":
    w = GridworldEnv()
    w.reset()
    w.step([1, 0])
    print(w)