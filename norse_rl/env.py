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
        0       Distance to food          0                       ~14
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

    MAX_SIZE = 500
    DIST_SCALE = 4 / math.sqrt(MAX_SIZE ** 2 + MAX_SIZE ** 2)

    action_labels = ["Left", "Right"]
    observation_labels = ["Distance", "Angle"]

    action_space = spaces.Box(0, 1, shape=(2,))
    observation_space = spaces.MultiDiscrete([MAX_SIZE, MAX_SIZE])
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 30}
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 50}
    pixel_scale = 5

    def __init__(self, food_items: int = 10, dt: float = 2.0):
        assert food_items < self.MAX_SIZE, f"Food must be < {self.MAX_SIZE}"
        self.food_items = food_items
        self.dt = dt

    def _draw_square(self, img, x, y, color):
        x = int(x)
        y = int(y)
        img[
            y - self.pixel_scale : y + self.pixel_scale,
            x - self.pixel_scale : x + self.pixel_scale,
        ] = color
        return img

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
        if dist < 0.5 * self.DIST_SCALE:
            self.food.remove(food_pos)  # Delete food
            reward = 1
            dist, food_pos = self._closest_food(self.state[:2])
        else:
            reward = 0

        # Define angle to food
        x = food_pos[0]-self.state[0]
        y = food_pos[1]-self.state[1]
        angle = math.atan2(y,x)-self.state[2]
        if angle > math.pi:
            angle = angle - 2*math.pi
        elif angle < -math.pi:
            angle = angle + 2*math.pi

        return np.array([dist, angle]), reward

    def render(self, mode="rgb_array"):
        img = np.zeros((*[x for x in self.observation_space.nvec], 3))
        if len(self.food) == 0:
            return img

        # Draw food
        for (x, y) in self.food:
            self._draw_square(img, x, y, [0, 1, 0])
        # Draw agent
        self._draw_square(img, *self.state[:2], [1, 0, 0])
        return img

    def reset(self):
        # Init in center pointing east
        self.state = np.array([x // 2 for x in self.observation_space.nvec] + [0])
        self._distribute_food()
        return self._observe()[0]

    def step(self, action):
        left_move, right_move = np.array(action).clip(-1, 1)
        angle = self.state[-1]

        d_rotation = (right_move - left_move) * self.dt / math.pi * 2
        d_x = (
            min(left_move, right_move) * math.cos(d_rotation) * self.dt
            + max(left_move, right_move) * math.cos(angle) * self.dt
        )
        d_y = (
            -min(left_move, right_move) * math.sin(d_rotation) * self.dt
            - max(left_move, right_move) * math.sin(angle) * self.dt
        )

        # if left_move and not right_move:
        #     self.state = np.array([*self.state[:2], angle + math.pi / 2 * self.dt])
        # elif right_move and not left_move:
        #     self.state = np.array([*self.state[:2], angle - math.pi / 2 * self.dt])
        # elif left_move and right_move:
        #     move = np.array([math.cos(angle), math.sin(angle)])
        #     # Ignore moves if angle is not aligned with axes
        #     if move.sum() > 0.5:
        #         self.state = np.array([*(self.state[:2] + move), angle])

        # Set new state and validate
        location = (self.state[:2] + np.array([d_x, d_y])).clip(0, self.MAX_SIZE)
        
        new_angle = angle + d_rotation

        if new_angle >= math.pi:
            # print("%f to %f" %(new_angle*180/math.pi, (new_angle - 2*math.pi)*180/math.pi))
            new_angle = new_angle - 2*math.pi
        if new_angle <= -math.pi:
            # print("%f to %f" %(new_angle*180/math.pi, (new_angle + 2*math.pi)*180/math.pi))
            new_angle = new_angle + 2*math.pi

        self.state = np.array([*location, new_angle])

        # Define observation
        observation, reward = self._observe()
        return observation, reward, len(self.food) == 0, {}


if __name__ == "__main__":
    w = GridworldEnv()
    w.reset()
    w.step([1, 0])
    print(w)
