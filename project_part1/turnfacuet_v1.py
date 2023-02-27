import gym
import gym.spaces as spaces
from tqdm.notebook import tqdm
import numpy as np
import mani_skill2.envs
import matplotlib.pyplot as plt
import torch.nn as nn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

from mani_skill2.utils.wrappers import RecordEpisode
from stable_baselines3 import SAC

class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps: int) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self._max_episode_steps = max_episode_steps

    def reset(self):
        self._elapsed_steps = 0
        return super().reset()

    def step(self, action):
        ob, rew, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        else:
            done = False
            info["TimeLimit.truncated"] = False
        return ob, rew, done, info

# A simple wrapper that adds a is_success key which SB3 tracks
class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, done, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, done, info
    
# define an SB3 style make_env function for evaluation
def make_env(env_id: str, max_episode_steps: int = None, record_dir: str = None):
    def _init() -> gym.Env:
        # NOTE: Import envs here so that they are registered with gym in subprocesses
        import mani_skill2.envs
        env = gym.make(env_id, obs_mode=obs_mode, reward_mode=reward_mode, control_mode=control_mode,)
        # For training, we regard the task as a continuous task with infinite horizon.
        # you can use the ContinuousTaskWrapper here for that
        if max_episode_steps is not None:
            env = ContinuousTaskWrapper(env, max_episode_steps)
        if record_dir is not None:
            env = SuccessInfoWrapper(env)
            env = RecordEpisode(
                env, record_dir, info_on_video=True, render_mode="cameras"
            )
        return env
    return _init

if __name__=="__main__":
    num_envs = 35 # you can increases this and decrease the n_steps parameter if you have more cores to speed up training
    env_id = "TurnFaucet-v1" #"LiftCube-v1"
    obs_mode = "state"
    control_mode = "pd_ee_delta_pose"
    reward_mode = "dense"

    # create one eval environment
    eval_env = SubprocVecEnv([make_env(env_id, record_dir="logs/"+env_id+"_videos") for i in range(1)])
    eval_env = VecMonitor(eval_env) # attach this so SB3 can log reward metrics
    eval_env.seed(0)
    eval_env.reset()

    # create num_envs training environments
    # we also specify max_episode_steps=100 to speed up training
    env = SubprocVecEnv([make_env(env_id, max_episode_steps=100) for i in range(num_envs)])
    env = VecMonitor(env)
    env.seed(0)
    obs = env.reset()

    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                            log_path="./logs/", eval_freq=32000,
                            deterministic=True, render=False) 

    checkpoint_callback = CheckpointCallback(
        save_freq=32000,
        save_path="./logs/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    set_random_seed(0) # set SB3's global seed to 0
    rollout_steps = 3200 #(12,6400) #3200

    # create our model
    policy_kwargs = dict(net_arch=[256, 256])
    model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
        batch_size=4096,
        tensorboard_log="./logs",
        learning_rate=3e-4,
        ent_coef = "0.08", 
        learning_starts = 10000,
        gamma=0.9)

    # Train with SAC
    model.learn(6_000_000, callback=[checkpoint_callback, eval_callback])
    model.save("./logs/latest_model")

    # optionally load back the model that was saved
    model = model.load("./logs/latest_model")

    eval_env.close() # close the old eval env
    # make a new one that saves to a different directory
    eval_env = SubprocVecEnv([make_env(env_id, record_dir="logs/"+env_id+"_eval_videos") for i in range(1)])
    eval_env = VecMonitor(eval_env) # attach this so SB3 can log reward metrics
    eval_env.seed(1)
    eval_env.reset()

    returns, ep_lens = evaluate_policy(model, eval_env, deterministic=True, render=False, return_episode_rewards=True, n_eval_episodes=10)
    success = np.array(ep_lens) < 200 # episode length < 200 means we solved the task before time ran out
    success_rate = success.mean()
    print(f"Success Rate: {success_rate}")
    print(f"Episode Lengths: {ep_lens}")