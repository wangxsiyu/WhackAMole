from gym import spaces
import numpy as np
import pygame

class Mole(spaces.Box):
    def __init__(self, low, high, shape, window_size):
        super().__init__(low = low, high = high, shape = shape)
        self.window_size = window_size 
        self.set_task_parameters()     
        self.reset()

    def step(self): 
        self.params['p_popping'] = 0.1
        if self.mole_life > 0:
            if np.random.random() < self.params['p_countdown']:
                self.mole_life -= 1
        elif self.mole_life == 0:
            if np.random.random() < self.params['p_popping']:
                self.reset()
            else:
                self._mole_location = np.array((0,0))

    def obs(self):
        return {"xy": self._mole_location}
    
    def reset(self):
        ps = self.params['locations']
        self.mole_life = 20
        i = np.random.randint(len(ps))
        self._mole_location = np.array(ps[i])

    def set_task_parameters(self, params = None):
        if params is None:
            params = dict()
            params['p_countdown'] = 0.9
            params['radius'] = 30
            wh = np.array(self.window_size)
            params['locations'] = np.array([[0.1, 0.1] * wh,[0.1, 0.9] * wh,[0.9, 0.1] * wh,[0.9, 0.9] * wh])
        self.params = params

    def get_task_parameters(self):
        return self.params

    def _render_frame(self, canvas):
        if all(self._mole_location != np.array((0,0))):
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                self._mole_location,
                self.params["radius"],
            )