import gym
from gym import spaces
import numpy as np
import math
from gym.utils.renderer import Renderer
from gym_lookamole.envs.mole import Mole
from gym_lookamole.envs.gaze import Gaze
import pygame

class LookAMole(gym.Env):
    metadata = {'render_modes': ["human", "rgb_array", "single_rgb_array"], "render_fps": 20}
    params = dict()
    def __init__(self, render_mode = None, window_size = (512, 512), render_fps = 20, n_frame_per_episode = 500):
        print(f'render mode: {render_mode}')
        self.window_size = window_size # PyGame window size
        self.metadata['render_fps'] = render_fps
        self.total_num_of_frames = n_frame_per_episode

        self._version_rotation_ismatch = None
        vMAX = 999.0
        low = np.array([-vMAX, -vMAX, -vMAX]).astype(np.float32)
        high = np.array([vMAX, vMAX, vMAX]).astype(np.float32)
        self.observation_space = spaces.Box(low, high)
        self.my_observation_space = spaces.Dict(
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
        self.n_actions = 3
        self.action_space = spaces.Discrete(self.n_actions)
        self.get_task_parameters()
        self.setup_rendermode(render_mode)

    def setup_rendermode(self, render_mode = None):
        self.render_mode = render_mode
        if self.render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render
            pygame.init()
            # pygame.font.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
        else:
            self.window = None
            self.clock = None
        self.renderer = Renderer(self.render_mode, self._render_frame)

    def set_params(self, params):
        self.params = params
        self.my_observation_space['mole'].set_task_parameters(params['mole'])
        self.my_observation_space['gaze'].set_task_parameters(params['gaze'])

    def get_task_parameters(self):
        params = dict()
        params["mole"] = self.my_observation_space['mole'].get_task_parameters()
        params["gaze"] = self.my_observation_space['gaze'].get_task_parameters()
        params['reward_rotation'] = 1
        params['epsilon_phi'] = math.pi/18
        self.params = params

    def _render_frame(self, mode: str):
        # This will be the function called by the Renderer to collect a single frame.
        assert mode is not None  # The renderer will not call this function with no-rendering.
        import pygame # avoid global pygame dependency. This method is not called with no-render.
    
        canvas = pygame.Surface(self.window_size)
        colval = 255
        canvas.fill((colval, colval, colval))
       
        self.my_observation_space['mole']._render_frame(canvas)
        if self._version_rotation_ismatch:
            width_line = 5
        else:
            width_line = 1

        self.my_observation_space['gaze']._render_frame(canvas, width_line)

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

    def calculate_phi(self, x, y):
            if x == 0:
                phi = math.pi/2 if y > 0 else math.pi/2 + math.pi
            else:
                phi = np.arctan(y/x)
                if x < 0:
                    phi += math.pi
            while phi < 0:
                phi += 2 * math.pi
            while phi >= math.pi * 2:
                phi -= 2 * math.pi
            return phi

    def is_match_phi(self, xy1, phi, xy2):
        phi2 = self.calculate_phi(xy2[0]-xy1[0], xy2[1]-xy1[1])
        if np.abs(phi - phi2) < self.params['epsilon_phi']:
            return True
        else:
            return False

    def step(self, action):
        reward = self.my_observation_space["gaze"].step(action)
        self.my_observation_space["mole"].step()

        self._version_rotation_ismatch = self.is_match_phi(self.my_observation_space["gaze"].obs()['xy'],
            self.my_observation_space["gaze"].obs()['phi'],
            self.my_observation_space["mole"].obs()['xy'])
        if self._version_rotation_ismatch:
            reward += self.params['reward_rotation']
        else:
            reward += 0

        self.frame_count += 1
        if self.frame_count >= self.total_num_of_frames:
            done = True
        else:
            done = False

        self.renderer.render_step()

        obs = self._get_obs()
        info = self._get_info()
        self.reward = self.reward + reward
        return obs, reward, done, info

    def num_actions(self):
        return self.n_actions

    def render(self):
        # Just return the list of render frames collected by the Renderer.
        return self.renderer.get_renders()

    def reset(self, seed = None, return_info = False):
        super().reset(seed = seed)
        self.frame_count = 0
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
        obs = [gaze['phi'], mole['xy']]
        return np.hstack(obs)

    def _get_obs(self):
        obs =  {"mole": self.my_observation_space['mole'].obs(), "gaze": self.my_observation_space['gaze'].obs()}
        return self.obs2vec(obs)

    def _get_info(self):
        return {'total-reward': self.reward}

    def close(self):
        if self.window is not None:
            import pygame 
            pygame.display.quit()
            pygame.quit()