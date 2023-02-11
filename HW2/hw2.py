import gym
from tqdm.notebook import tqdm
import numpy as np
import mani_skill2.envs
import matplotlib.pyplot as plt
import sapien.core as sapien
import mplib
import numpy as np
from sapien.utils.viewer import Viewer

from mani_skill2.utils.wrappers import RecordEpisode
from IPython.display import Video
def play(env, policy, steps=100):
    # reset environment to a clean state
    obs = env.reset()

    for i in tqdm(range(steps)):
        # repeatedly sample actions from the policy function
        action = policy(obs)
        # step through the environment and save the new observation, reward, done, and additional information
        obs, reward, done, info = env.step(action)
        if not IN_COLAB: env.render() # will render with a window if possible
        if done:
            # whenever an env is done, either we have succeeded or 
            # perhaps have entered a failed state. Any case, we will reset the env
            # Failure states usually mean we reached a time_limit (default is 200 steps here)
            # or the robot has entered some irrecoverable state that the environemnt defines as needing a reset
            obs = env.reset()


env_id = "LiftCube-v0"
# create environment
env = gym.make(env_id)
# for Colab users we wrap an environment wrapper to auto save videos, no need to learn how RecordEpisode works
env = RecordEpisode(env, "./videos", render_mode="cameras", info_on_video=True)
def policy(obs):
    action = np.zeros(env.action_space.shape)
    action[7] = 1
    return action
play(env, policy, steps=100)

# Save the video
env.flush_video()
# close the environment and release resources
env.close()
Video("./videos/0.mp4", embed=True) # Watch our replay