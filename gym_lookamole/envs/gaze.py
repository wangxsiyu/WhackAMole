from gym import spaces
import math
import numpy as np
import pygame

class Gaze(spaces.Box):
    def __init__(self, low, high, shape, window_size):
        super().__init__(low = low, high = high, shape = shape)
        self.window_size = window_size
        self._gaze_location = np.array(window_size)/2
        self.set_task_parameters()
        self.reset()

    def set_task_parameters(self, params = None):
        if params is None:
            params = dict()
            params['radius'] = 100
            params['_v0'] = math.pi/18
            params['_cost_turn'] = 0.1
        self.params = params

    def step(self, action):
        if action == 1:
            self._vphi = self.params['_v0']
            reward = -0.1 #self.params['_cost_turn']
        elif action == 2:
            self._vphi = -self.params['_v0']
            reward = -0.1 #self.params['_cost_turn']
        else:
            self._vphi = 0
            reward = 0
        self.phi = self.regularize_phi(self.phi + self._vphi)
        return reward

    def regularize_phi(self, x):
        while x >= 2 * math.pi: # keep phi between 0 to 2 pi
            x -= 2 * math.pi
        while x < 0:
            x += 2 * math.pi
        return x

    def reset(self):
        self.phi = self.regularize_phi(np.random.random() * 2 * math.pi)

    def obs(self):
        return {"phi": self.phi, "xy": self._gaze_location}

    def get_task_parameters(self):
        return self.params

    def get_xy_front(self):
        x, y = self._gaze_location
        x = np.cos(self.phi) * self.params['radius'] + x
        y = np.sin(self.phi) * self.params['radius'] + y
        return np.append(x, y)

    def _render_frame(self, canvas, width_line = 1):
        col_gaze = (255, 0, 0)
        pygame.draw.circle(
                canvas,
                col_gaze,
                self._gaze_location,
                self.params['radius'],
                width = 1
            )
        pygame.draw.line(
                canvas, 
                col_gaze, 
                self._gaze_location, 
                self.get_xy_front(), 
                width = width_line
            )

        