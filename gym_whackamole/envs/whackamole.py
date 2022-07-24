import gym
from gym import spaces
import math
import numpy as np
from gym.utils.renderer import Renderer
import pygame

class Mole(spaces.Box):
    def __init__(self, low, high, shape, window_size):
        super().__init__(low = low, high = high, shape = shape)
        self.window_size = window_size
        self.is_visible = 0
        self.p_popping = 0.1
        self.max_life = 5
        self.radius = 10
        self.reward_hit = 100
        self.reward_miss = -10
        self.reset()

    def collide(self, mole, gaze):
        xy_mole = mole["xy"]
        r_mole = mole["radius"]
        xy_gaze = mole["xy"]
        r_gaze = mole["radius"]
        dis = np.sqrt(np.sum((xy_gaze - xy_mole) ** 2))
        if dis < np.abs(r_gaze - r_mole):
            return True
        else:
            return False

    def is_mole_in_gaze(self, gaze):
        did_i_die = self.collide(self.obs(),gaze)
        if did_i_die:
            self.die()
        return did_i_die

    def step(self, gaze, action_hit): 
        if self._mole_life > 0:
            self._mole_life -= 1
            if self._mole_life == 0:
                self.die()
        elif np.random.random() < self.p_popping:
            self.pop()

        if action_hit == 1:
            if self.is_mole_in_gaze(gaze):
                reward = self.reward_hit
            else:
                reward = self.reward_miss
        else:
            reward = 0
        return reward

    def pop(self):
        self._mole_life = self.max_life
        self.is_visible = 1
        tx, ty = self.sample_pos()
        self.set_pos(tx, ty)

    def sample_pos(self):
        t = np.random.random(size = 2) * self.window_size
        return t[0], t[1]

    def die(self):
        self.is_visible = 0
        self.set_pos(0,0)

    def set_pos(self, x, y):
        self._mole_location = np.array([x,y])

    def reset(self):
        self._mole_life = 0
        self.die()

    def obs(self):
        return {"xy": self._mole_location * self.is_visible, "radius": self.radius, "isvisible": self.is_visible}

class Gaze(spaces.Box):
    def __init__(self, low, high, shape, window_size):
        super().__init__(low = low, high = high, shape = shape)
        self.window_size = window_size
        self.radius = 50
        self.alpha_gaze = 2
        self.alpha_dir = 2
        self.cost_action_step = -2
        self.cost_action_dir = -1
        self._gaze_velosity_initial = 1.0
        self._gaze_velosity_phi_initial = math.pi/30
        self.reset()

    def step(self, action_step, action_dir):
        self._gaze_location = self._gaze_location
        r1 = self.accelerate_step(action_step)
        r2 = self.accelerate_dir(action_dir)
        self.move_gaze()
        reward = r1 + r2
        return reward

    def accelerate_step(self, action_step):
        if action_step == 0: # no action taken
            reward = 0
        else:
            reward = self.cost_action_step

        if action_step == 1: # accelarate
            if self._gaze_velosity == 0:
                self._gaze_velosity = self._gaze_velosity_initial
            else:
                self._gaze_velosity = self._gaze_velosity * self.alpha_gaze
        elif action_step == 2: # decelarate
            self._gaze_velosity = self._gaze_velosity / self.alpha_gaze
            if self._gaze_velosity < self._gaze_velosity_initial:
                self._gaze_velosity = 0

        return reward

    def accelerate_dir(self, action_dir):
        if action_dir == 0: # no action taken
            reward = 0
        else:
            reward = self.cost_action_dir

        if action_dir == 1: # accelarate
            if self._phi_velosity == 0:
                self._phi_velosity = self._gaze_velosity_phi_initial * np.sign(np.random.random() - 0.5)
            else:
                self._phi_velosity = self._phi_velosity * self.alpha_dir
        elif action_dir == 2: # decelarate
            self._phi_velosity = self._phi_velosity / self.alpha_dir
            if np.abs(self._phi_velosity) < self._gaze_velosity_phi_initial:
                self._phi_velosity = 0
        elif action_dir == 3: # change direction
            if self._phi_velosity > 0:
                self._phi_velosity = -self._gaze_velosity_phi_initial
            elif self._phi_velosity < 0:
                self._phi_velosity = self._gaze_velosity_phi_initial
        return reward

    def move_gaze(self):
        phi = self.phi + self._phi_velosity
        x, y = self._gaze_location
        x += np.cos(phi) * self._gaze_velosity
        y += np.sin(phi) * self._gaze_velosity
        self.set_pos(x, y)

    def calculate_phi(self):
        x, y = self._gaze_location
        if x == 0:
            phi = math.pi/2 if y > 0 else math.pi/2 + math.pi
        else:
            phi = np.arctan(y/x)
            if x < 0:
                phi += math.pi
        self.phi = phi

    def sample_pos(self):
        t = np.random.random(size = 2) * self.window_size
        return t[0], t[1]

    def is_valid_xy(self, x, y):
        if x > 0 and x < self.window_size[0] and y > 0 and y < self.window_size[1]:
            return True
        else:
            return False

    def set_pos(self, x, y):
        x = float(x)
        y = float(y)
        if self.is_valid_xy(x,y):
            self._gaze_location = np.array([x,y])
            self.calculate_phi()
        else: 
            self.reset()

    def reset(self):
        tx, ty = self.sample_pos()
        self.set_pos(tx, ty)
        self._gaze_velosity = np.array([0])
        self._phi_velosity = np.array([0])

    def obs(self):
        return {"xy":self._gaze_location, "radius": self.radius, 
                "v_step": self._gaze_velosity, "v_phi": self._phi_velosity}

class WhackAMole(gym.Env):
    metadata = {'render_modes': ["human", "rgb_array", "single_rgb_array"], "render_fps": 5}
    def __init__(self, render_mode = None):
        print(f'render mode: {render_mode}')
        self.render_mode = render_mode
        self.window_size = (512, 512) # PyGame window size
        self.total_num_of_frames = 1000
        self.action_space = spaces.Dict(
            {
                "gaze_dir": spaces.Discrete(4),
                "gaze_step": spaces.Discrete(3),
                "hit": spaces.Discrete(2)
            }
        )
        self.observation_space = spaces.Dict(
            {
                "mole": Mole(low = np.array([0, self.window_size[0]]),
                             high = np.array([0, self.window_size[1]]),
                             shape = (2,),
                             window_size = self.window_size),
                "gaze": Gaze(low = np.array([0, self.window_size[0]]),
                             high = np.array([0, self.window_size[1]]),
                             shape = (2,),
                             window_size = self.window_size)
            }
        )
        if self.render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
        else:
            self.window = None
            self.clock = None
        self.renderer = Renderer(self.render_mode, self._render_frame)

    def step(self, action):
        r1 = self.observation_space["gaze"].step(action["gaze_step"],action["gaze_dir"])
        r2 = self.observation_space["mole"].step(self.observation_space["gaze"].obs(), action["hit"])
        self.reward = self.reward + r1 + r2
        obs = self._get_obs()
        info = self._get_info()

        self.frame_count -= 1
        if self.frame_count == 0:
            done = True
        else:
            done = False

        # add a frame to the render collection
        self.renderer.render_step()

        return obs, self.reward, done, info

    def render(self):
        # Just return the list of render frames collected by the Renderer.
        return self.renderer.get_renders()

    def reset(self, seed = None, return_info = False):
        super().seed(seed)
        self.frame_count = 0;
        self.reward = 0
        self.observation_space["mole"].reset()
        self.observation_space["gaze"].reset()
        obs = self._get_obs()
        info = self._get_info()
        # clean the render collection and add the initial frame
        self.renderer.reset()
        self.renderer.render_step()
        return obs, info if return_info else obs

    def _get_obs(self):
        return {"mole": self.observation_space['mole'].obs(), "gaze": self.observation_space['gaze'].obs()}

    def _get_info(self):
        return {'reward': self.reward}

    def close(self):
        if self.window is not None:
            import pygame 
            pygame.display.quit()
            pygame.quit()

    def _render_frame(self, mode: str):
        # This will be the function called by the Renderer to collect a single frame.
        assert mode is not None  # The renderer will not call this function with no-rendering.
        import pygame # avoid global pygame dependency. This method is not called with no-render.
    
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        pygame.display.set_caption(f"Reward = {self.reward}")

        now_mole = self.observation_space["mole"].obs()
        if now_mole["isvisible"] == 1:
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                now_mole["xy"],
                now_mole["radius"],
            )

        now_gaze = self.observation_space["gaze"].obs()
        pygame.draw.circle(
                canvas,
                (255, 0, 0),
                now_gaze["xy"],
                now_gaze["radius"],
                width = 1
            )

        if mode == "human":
            assert self.window is not None
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array or single_rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
