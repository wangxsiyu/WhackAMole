import gym
from gym import spaces
import numpy as np
from gym.utils.renderer import Renderer
from gym_whackamole.envs.mole import Mole
from gym_whackamole.envs.gaze import Gaze
import pygame

class WhackAMole(gym.Env):
    metadata = {'render_modes': ["human", "rgb_array", "single_rgb_array"], "render_fps": 20}
    params = dict()
    def __init__(self, render_mode = None, window_size = (512, 512), render_fps = 20, n_frame_per_episode = 500):
        print(f'render mode: {render_mode}')
        self.render_mode = render_mode
        self.window_size = window_size # PyGame window size
        self.metadata['render_fps'] = render_fps
        self.total_num_of_frames = n_frame_per_episode
        # self.action_space = spaces.Dict(
        #     {
        #         "gaze_dir": spaces.Discrete(4),
        #         "gaze_step": spaces.Discrete(3),
        #         "hit": spaces.Discrete(2)
        #     }
        # )
        self.action_space = spaces.Discrete(7)
        vMAX = 999.0
        # x,y, radius,is_visible, is_hit (mole), x, y, phi, radius, v_step, v_dir (gaze)
        low = np.array([0,0,0,0,0,
            0,0,-vMAX, 0,0,-vMAX]).astype(np.float32)
        high = np.array([self.window_size[0],self.window_size[1],vMAX,1,1,
            self.window_size[0],self.window_size[1], vMAX, vMAX, vMAX, vMAX]).astype(np.float32)
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
        self.get_task_parameters()

    def set_params(self, params):
        self.params = params
        self.my_observation_space['mole'].set_task_parameters(params['mole'])
        self.my_observation_space['gaze'].set_task_parameters(params['gaze'])

    def get_task_parameters(self):
        params = dict()
        params["mole"] = self.my_observation_space['mole'].get_task_parameters()
        params["gaze"] = self.my_observation_space['gaze'].get_task_parameters()
        self.params = params

    def _render_frame(self, mode: str):
        # This will be the function called by the Renderer to collect a single frame.
        assert mode is not None  # The renderer will not call this function with no-rendering.
        import pygame # avoid global pygame dependency. This method is not called with no-render.
    
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
       
        self.my_observation_space['mole']._render_frame(canvas)
        ishit = self.my_observation_space['mole'].obs()['ishit']
        self.my_observation_space['gaze']._render_frame(canvas, ishit)
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

    def step(self, action):
        action = self.action_transform(action)
        r1 = self.my_observation_space["gaze"].step(action["gaze_step"],action["gaze_dir"])
        r2 = self.my_observation_space["mole"].step(self.my_observation_space["gaze"].obs(), action["hit"])
        reward = r1 + r2
        self.reward = self.reward + reward

        self.frame_count += 1
        if self.frame_count <= self.total_num_of_frames:
            done = False
        else:
            done = True

        # add a frame to the render collection
        self.renderer.render_step()

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, done, info

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
        return {'total-reward': self.reward}

    def close(self):
        if self.window is not None:
            import pygame 
            pygame.display.quit()
            pygame.quit()