import gym
from gym import spaces
import math
import numpy as np
from gym.utils.renderer import Renderer
import pygame

class Mole(spaces.Box):
    def __init__(self, low, high, shape, window_size, mode = None):
        super().__init__(low = low, high = high, shape = shape)
        self.window_size = window_size
        self.is_visible = 0
        self.p_popping = 0.2
        self.max_life = 20
        self.radius = 10
        self.reward_hit = 100
        self.reward_miss = -10
        self.mode_mole = mode
        self.reset()

    def collide(self, mole, gaze):
        xy_mole = mole["xy"]
        r_mole = mole["radius"]
        xy_gaze = gaze["xy"]
        r_gaze = gaze["radius"]
        dis = np.sqrt(np.sum((xy_gaze - xy_mole) ** 2))
        if dis < np.abs(r_gaze + r_mole): # as long as it touches
            # print("hit")
            return True
        else:
            return False

    def step(self, gaze, action_hit): 
        if self.am_I_hit == 1:
            self.die()
        else:
            if self._mole_life > 0:
                self._mole_life -= 1
                if self._mole_life == 0:
                    self.die()
            elif np.random.random() < self.p_popping:
                self.pop()

        self.am_I_hit = 0
        if action_hit == 1:
            if self.is_visible == 1 and self.collide(self.obs(),gaze):
                self.am_I_hit = 1
                reward = self.reward_hit
            else:
                self.am_I_hit = -1
                reward = self.reward_miss
        else:
            reward = 0
        return reward

    def pop(self):
        if self.mode_mole == "maxlife":
            self._mole_life = self.max_life * 10
        else:
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
        self.am_I_hit = 0
        self._mole_life = 0
        self.die()

    def obs(self):
        return {"xy": self._mole_location * self.is_visible, "radius": self.radius, "isvisible": self.is_visible, "ishit": self.am_I_hit}

class Gaze(spaces.Box):
    def __init__(self, low, high, shape, window_size):
        super().__init__(low = low, high = high, shape = shape)
        self.window_size = window_size
        self.radius = 50
        self.alpha_gaze = 2
        self.alpha_dir = 2
        self._gaze_velosity_initial = 1.0
        self._gaze_velosity_phi_initial = math.pi/30
        self.cost_action_step = 0
        self.cost_action_dir = 0
        self.punish_de_velosity_at_0 = -2
        self.punish_de_phi_at_0 = -1
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
            if self._gaze_velosity == 0:
                reward += self.punish_de_velosity_at_0
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
            if self._phi_velosity == 0:
                reward += self.punish_de_phi_at_0
            self._phi_velosity = self._phi_velosity / self.alpha_dir
            if np.abs(self._phi_velosity) < self._gaze_velosity_phi_initial:
                self._phi_velosity = 0
        elif action_dir == 3: # change direction
            if self._phi_velosity == 0:
                reward += self.punish_de_phi_at_0
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
    metadata = {'render_modes': ["human", "rgb_array", "single_rgb_array"], "render_fps": 20}
    def __init__(self, render_mode = None, mode_mole = None):
        print(f'render mode: {render_mode}')
        self.render_mode = render_mode
        self.window_size = (512, 512) # PyGame window size
        self.total_num_of_frames = 500
        # self.action_space = spaces.Dict(
        #     {
        #         "gaze_dir": spaces.Discrete(4),
        #         "gaze_step": spaces.Discrete(3),
        #         "hit": spaces.Discrete(2)
        #     }
        # )
        self.action_space = spaces.Discrete(7)
        vMAX = 99999.0
        low = np.array([0,0,0,0,0,0,0,0,-vMAX,-vMAX]).astype(np.float32)
        high = np.array([self.window_size[0],self.window_size[1],vMAX,1,1,
            self.window_size[0],self.window_size[1], vMAX, vMAX, vMAX]).astype(np.float32)
        self.observation_space = spaces.Box(low, high)

        self.my_observation_space = spaces.Dict(
            {
                "mole": Mole(low = np.array([0, self.window_size[0]]),
                             high = np.array([0, self.window_size[1]]),
                             shape = (2,),
                             window_size = self.window_size,
                             mode = mode_mole),
                "gaze": Gaze(low = np.array([0, self.window_size[0]]),
                             high = np.array([0, self.window_size[1]]),
                             shape = (2,),
                             window_size = self.window_size)
            }
        )
        if self.render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render
            pygame.init()
            pygame.font.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
        else:
            self.window = None
            self.clock = None
        self.renderer = Renderer(self.render_mode, self._render_frame)

    def action_transform(self, action):
        a = spaces.Dict(
            {
                "gaze_dir": spaces.Discrete(4),
                "gaze_step": spaces.Discrete(3),
                "hit": spaces.Discrete(2)
            }
        )
        if action == 1:
            a["hit"] = 1
        else:
            a["hit"] = 0
        if action >= 2 and action <= 3:
            a["gaze_step"] = action - 1
        else:
            a["gaze_step"] = 0
        if action >= 4 and action <= 6:
            a["gaze_dir"] = action - 3
        else:
            a["gaze_dir"] = 0
        return(a)

    def step(self, action):
        action = self.action_transform(action)
        r1 = self.my_observation_space["gaze"].step(action["gaze_step"],action["gaze_dir"])
        r2 = self.my_observation_space["mole"].step(self.my_observation_space["gaze"].obs(), action["hit"])
        self.reward = self.reward + r1 + r2

        self.frame_count += 1
        if self.frame_count <= self.total_num_of_frames:
            done = False
        else:
            done = True

        # add a frame to the render collection
        self.renderer.render_step()

        obs = self._get_obs()
        info = self._get_info()
        return obs, self.reward, done, info

    def render(self):
        # Just return the list of render frames collected by the Renderer.
        return self.renderer.get_renders()

    def reset(self, seed = None, return_info = False):
        super().reset(seed = seed)
        self.frame_count = 0;
        self.reward = 0
        self.my_observation_space["mole"].reset()
        self.my_observation_space["gaze"].reset()
        # clean the render collection and add the initial frame
        self.renderer.reset()
        self.renderer.render_step()

        obs = self._get_obs()
        info = self._get_info()
        return obs if not return_info else (obs, info)

    def obs2vec(self, obs):
        mole = obs["mole"]
        gaze = obs["gaze"]
        obs = [mole["xy"], mole["radius"], mole["isvisible"],mole["ishit"], gaze["xy"], gaze["radius"], gaze["v_step"], gaze["v_phi"]]
        return np.hstack(obs)

    def _get_obs(self):
        obs =  {"mole": self.my_observation_space['mole'].obs(), "gaze": self.my_observation_space['gaze'].obs()}
        return self.obs2vec(obs)

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
        now_mole = self.my_observation_space["mole"].obs()
        if now_mole["isvisible"] == 1:
            if now_mole["ishit"] == 1:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 255),
                    now_mole["xy"],
                    now_mole["radius"]*2,
                )
            else:
                pygame.draw.circle(
                    canvas,
                    (0, 255, 0),
                    now_mole["xy"],
                    now_mole["radius"],
                )

        now_gaze = self.my_observation_space["gaze"].obs()
        if now_mole["ishit"] == -1:
            width_gaze = 0
            col_gaze = (255, 0, 0)
        elif now_mole["ishit"] == 1:
            width_gaze = 5
            col_gaze = (255, 0, 0)
        else:
            width_gaze = 1
            col_gaze = (255, 0, 0)

        pygame.draw.circle(
                canvas,
                col_gaze,
                now_gaze["xy"],
                now_gaze["radius"],
                width = width_gaze
            )

        # font1 = pygame.font.SysFont("Garamond", 72)
        # canvas_text = font1.render(f"Reward = {self.reward}", True, (0,0,255))
        if mode == "human":
            assert self.window is not None
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            # self.window.blit(canvas_text, (1,1))
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array or single_rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
