import math
from pathlib import Path
import random
import time

import gym
from gym import spaces
import numpy as np
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
from ipywidgets import Image

import norse_rl.util as util


class MazeworldDistEnv(gym.Env):
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
    observation_labels = [
        "Left ∠",
        "Right ∠",
        "Right\nWhisker",
        "Left\nWhisker",
        "Distance",
    ]

    action_space = spaces.Box(0, 1, shape=(2,))
    observation_space = spaces.MultiDiscrete([MAX_SIZE, MAX_SIZE])
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 30}
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 50}
    pixel_scale = 5

    def __init__(
        self,
        food_items: int = 10,
        image_scale: float = 1.0,
        dt: float = 1.0,
        level: int = 1,
    ):
        assert food_items < self.MAX_SIZE, f"Food must be < {self.MAX_SIZE}"
        self.food_items = food_items
        self.image_scale = image_scale
        self.dt = dt
        self.imgMouse = mpimg.imread("images/Mouse_40px.png")
        self.imgCompleted = Image.from_file("images/Completed_250px.png")
        self.sizeMouse = int(self.imgMouse.shape[0])
        self.wall = []
        self.food = []
        self.level = level
        self.tileSize = 10
        
        # Set seed
        np.random.seed(0)

    def _draw_square(self, canvas, x, y, color, size):
        canvas.fill_style = color
        canvas.fill_rect(
            int((x - size / 2) * self.image_scale),
            int((y - size / 2) * self.image_scale),
            size,
        )
        return canvas

    def _draw_agent(self, canvas, x, y, color):

        imgRot = ndimage.rotate(
            (self.imgMouse * 255).astype("uint8"),
            self.state[2] * 180 / math.pi,
            reshape=False,
        )
        canvas.put_image_data(
            imgRot,
            self.state[0] * self.image_scale - self.sizeMouse / 2,
            self.state[1] * self.image_scale - self.sizeMouse / 2,
        )

        return canvas

    def _get_level(self):
        file1 = open("levels/level_" + str(self.level) + ".txt", "r")
        x = 0
        y = 0
        cheese = []

        while True:

            line = file1.readline()
            for x in range(len(line)):
                if line[x] == "#":
                    self.wall.append(
                        ((x + 0.5) * self.tileSize, (y + 0.5) * self.tileSize)
                    )

                if line[x] == "*":
                    self.food.append(
                        ((x + 0.5) * self.tileSize, (y + 0.5) * self.tileSize)
                    )

            y += 1
            if not line:
                break

        file1.close()
        self.food_items = len(self.food)

    def _draw_walls(self, canvas):
        for w in self.wall:
            self._draw_square(canvas, w[0], w[1], "rgb(153, 76, 0)", self.tileSize)

    def _distribute_food(self):
        indices = np.random.randint(
            self.sizeMouse, self.MAX_SIZE - self.sizeMouse, size=(self.food_items, 2)
        )
        self.food = indices.tolist()

    def _getAngle(self, food_pos):
        # Define angle to food
        x = food_pos[0] - self.state[0]
        y = food_pos[1] - self.state[1]
        target_angle = math.atan2(-y, x)
        current_angle = self.state[2]
        angle = math.atan2(
            math.sin(current_angle - target_angle),
            math.cos(current_angle - target_angle),
        )
        return angle

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
        if dist < self.DIST_SCALE * 20:  # Radius of 5
            self.food.remove(food_pos)  # Delete food
            reward = 1
            dist, food_pos = self._closest_food(self.state[:2])
        else:
            reward = 0

        # Define angle to food
        if len(self.food) == 0:
            angle = 0
        else:
            angle = self._getAngle(food_pos)
        angle_left = max(0, -angle) * self.ROTATION_SCALE
        angle_right = max(0, angle) * self.ROTATION_SCALE

        return (
            np.array([angle_left, angle_right, self.state[3], self.state[4], dist]),
            reward,
        )

    def render(self, canvas):
        # Draw background
        canvas.fill_style = "rgb(50, 50, 50)"
        canvas.fill_rect(0, 0, 400, 400)

        if len(self.food) == 0:
            canvas.draw_image(
                self.imgCompleted, self.MAX_SIZE / 2 - 125, self.MAX_SIZE / 2 - 125
            )
        else:
            if len(self.food) == 0:
                return canvas

            # Draw food
            for (x, y) in self.food:
                self._draw_square(canvas, x, y, "rgb(246, 195, 53)", self.tileSize)

            # Draw Walls
            self._draw_walls(canvas)

            # Draw agent
            self._draw_agent(canvas, *self.state[:2], "red")
        return canvas

    def _check_collisions(self, d_x, d_y):
        new_x = self.state[0] + d_x
        new_y = self.state[1] + d_y
        max_val = 0.5
        inc_val = max_val / 1
        l_val = 0
        r_val = 0

        for w in self.wall:
            if abs(w[0] - self.state[0]) <= (self.tileSize + self.sizeMouse) / 2:
                if abs(w[1] - new_y) <= (self.tileSize + self.sizeMouse) / 2:
                    d_y = 0

                    r_val = inc_val
                    l_val = 0
                    if self.state[2] < math.pi / 2:
                        r_val = 0
                        l_val = inc_val
                    if self.state[2] < 0:
                        r_val = inc_val
                        l_val = 0
                    if self.state[2] < -math.pi / 2:
                        r_val = 0
                        l_val = inc_val

            if abs(w[1] - self.state[1]) <= (self.tileSize + self.sizeMouse) / 2:
                if abs(w[0] - new_x) <= (self.tileSize + self.sizeMouse) / 2:
                    d_x = 0

                    r_val = 0
                    l_val = inc_val
                    if self.state[2] < math.pi / 2:
                        r_val = inc_val
                        l_val = 0
                    if self.state[2] < 0:
                        r_val = 0
                        l_val = inc_val
                    if self.state[2] < -math.pi / 2:
                        r_val = inc_val
                        l_val = 0

        if r_val > 0:
            self.state[3] = min(max_val, self.state[3] + r_val)
        else:
            self.state[3] = 0

        if l_val > 0:
            self.state[4] = min(max_val, self.state[4] + l_val)
        else:
            self.state[4] = 0

        return d_x, d_y

    def reset(self):
        # Init in center pointing east
        self.state = np.array(
            [x // 2 for x in self.observation_space.nvec] + [0] + [0] + [0]
        )
        self._get_level()
        return self._observe()[0]

    def step(self, action, random_movement: bool = True):
        left_move, right_move = np.array(action).clip(0, 1)
        angle = self.state[2]

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

        d_x, d_y = self._check_collisions(d_x, d_y)

        # Set new state and validate
        location = (self.state[:2] + np.array([d_x, d_y])).clip(0, self.MAX_SIZE)

        whiskers = self.state[3:5]
        self.state = np.array([*location, new_angle, *whiskers])
        self.state = np.nan_to_num(self.state, 0)  # Remove NaN

        # Define observation
        is_done = len(self.food) == 0
        if is_done:
            observation, reward = np.array([0, 0, 0, 0]), 0
        else:
            observation, reward = self._observe()
        return observation, reward, is_done, {}
