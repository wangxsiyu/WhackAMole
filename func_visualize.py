import numpy as np
def visualize_env(tenv, model = None):
    observation, info = tenv.reset(return_info = True)
    # n_frames = 100
    total_reward = 0
    while True:
        tenv.render()
        # action = {"gaze_dir": np.random.random_integers(0,4), 
        #           "gaze_step": np.random.random_integers(0,3), 
        #           "hit": np.random.random_integers(0,2)}
        if model is None:
            action = np.random.random_integers(0, 7)
        else:
            action, _state = model.predict(observation, deterministic=False)
        observation, reward, done, info = tenv.step(action)
        total_reward += reward
        if done:
            break;
    print(total_reward)
    # print(info["total-reward"])
    tenv.close()