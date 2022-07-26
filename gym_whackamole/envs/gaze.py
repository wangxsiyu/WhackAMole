from gym import spaces
import math
import numpy as np
import pygame

class Gaze(spaces.Box):
    def __init__(self, low, high, shape, window_size):
        super().__init__(low = low, high = high, shape = shape)
        self.window_size = window_size
        self.set_task_parameters()
        self.reset()

    def set_task_parameters(self, params = None):
        if params is None:
            params = dict()
            params['radius'] = 50
            params['alpha_step'] = 2
            params['alpha_dir'] = 2
            params['_vstep_initial'] = 1.0
            params['_vstep_MAX'] = 30.0
            params['_vphi_initial'] = math.pi/18
            params['_vphi_MAX'] = math.pi/2
            params['punish_action_step'] = 0
            params['punish_action_dir'] = 0
            params['punish_de_step_at_0'] = -1
            params['punish_de_phi_at_0'] = -1
            params['punish_ac_step_at_MAX'] = -1
            params['punish_ac_phi_at_MAX'] = -1
            params['punish_outofbox'] = 0
            params['is_boundary_flip'] = 1
        self.params = params

    def accelerate_dir(self, action_dir):
        if action_dir == 0: # no action taken
            reward = 0
        else:
            reward = self.params['punish_action_dir']

        if action_dir == 1: # accelarate
            if np.abs(self._vphi) >= self.params['_vphi_MAX']:
                reward += self.params['punish_ac_phi_at_MAX']
            else:
                if self._vphi == 0:
                    self._vphi = self.params['_vphi_initial'] * np.sign(np.random.random() - 0.5)
                else:
                    self._vphi = self._vphi * self.params['alpha_dir']
                    if self._vphi > self.params['_vphi_MAX']:
                        self._vphi = self.params['_vphi_MAX']
                    elif self._vphi < -self.params['_vphi_MAX']:
                        self._vphi = -self.params['_vphi_MAX']
        elif action_dir == 2: # decelarate
            if self._vphi == 0:
                reward += self.params['punish_de_phi_at_0']
            else:
                self._vphi = self._vphi / self.params['alpha_dir']
                if np.abs(self._vphi) < self.params['_vphi_initial']:
                    self._vphi = 0
        elif action_dir == 3: # change direction
            if self._vphi == 0:
                reward += self.params['punish_de_phi_at_0']
            else:
                if self._vphi > 0:
                    self._vphi = -self.params['_vphi_initial']
                elif self._vphi < 0:
                    self._vphi = self.params['_vphi_initial']
        return reward

    def accelerate_step(self, action_step):
        if action_step == 0: # no action taken
            reward = 0
        else:
            reward = self.params['punish_action_step']

        if action_step == 1: # accelarate
            if self._vstep == self.params['_vstep_MAX']:
                reward += self.params['punish_ac_step_at_MAX']
            else:
                if self._vstep == 0:
                    self._vstep = self.params['_vstep_initial']
                else:
                    self._vstep = self._vstep * self.params['alpha_step']
                    if self._vstep > self.params['_vstep_MAX']:
                        self._vstep = self.params['_vstep_MAX']
        elif action_step == 2: # decelarate
            if self._vstep == 0: 
                reward += self.params['punish_de_step_at_0']
            else:
                self._vstep = self._vstep / self.params['alpha_step']
                if self._vstep < self.params['_vstep_initial']:
                    self._vstep = 0
        return reward

    def step(self, action_step, action_dir):
        r1 = self.accelerate_step(action_step)
        r2 = self.accelerate_dir(action_dir)
        r3 = self.move_gaze()
        reward = r1 + r2 + r3
        return reward

    def move_gaze(self):
        self.phi = self.phi + self._vphi
        x, y = self._gaze_location
        dx = np.cos(self.phi) * self._vstep
        dy = np.sin(self.phi) * self._vstep
        if self.is_valid_xy(x+dx,y+dy):
            x += dx
            y += dy
            reward = 0
        else: 
            if self.params['is_boundary_flip'] == 1: # flip direction
                self.phi = self.phi + math.pi
                self._vphi = 0
                self._vstep = 0
            reward = self.params['punish_outofbox']
        self.set_pos(x, y)
        return reward

    def is_valid_xy(self, x, y):
        if x > 0 and x < self.window_size[0] and y > 0 and y < self.window_size[1]:
            return True
        else:
            return False

    def sample_pos(self):
        t = np.random.random(size = 2) * self.window_size
        return t[0], t[1]

    def set_pos(self, x, y):
        x = float(x)
        y = float(y)
        self._gaze_location = np.array([x,y])

    def reset(self):
        tx, ty = self.sample_pos()
        self.set_pos(tx, ty)
        self.phi = np.random.random() * 2 * math.pi
        self._vstep = np.array([0])
        self._vphi = np.array([0])

    def obs(self):
        return {"xy":self._gaze_location, "phi": self.phi, "radius": self.params['radius'], 
                "v_step": self._vstep, "v_phi": self._vphi}

    def get_task_parameters(self):
        return self.params

    def get_xy_front(self):
        x, y = self._gaze_location
        x = np.cos(self.phi) * self.params['radius'] + x
        y = np.sin(self.phi) * self.params['radius'] + y
        return np.append(x, y)

    def _render_frame(self, canvas, ishit = 0):
        if ishit == -1:
            width_gaze = 0
        elif ishit == 1:
            width_gaze = 5
        else:
            width_gaze = 1
        col_gaze = (255, 0, 0)
        pygame.draw.circle(
                canvas,
                col_gaze,
                self._gaze_location,
                self.params['radius'],
                width = width_gaze
            )
        pygame.draw.line(canvas, col_gaze, self._gaze_location, self.get_xy_front())

        