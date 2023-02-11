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


def setup_planner(robot):
    link_names = [link.get_name() for link in robot.get_links()]
    joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    planner = mplib.Planner(
        urdf="/content/drive/MyDrive/Colab_Notebooks/panda_v2.urdf",
        srdf="/content/drive/MyDrive/Colab_Notebooks/panda_v2.srdf",
        user_link_names=link_names,
        user_joint_names=joint_names,
        move_group="panda_hand",
        joint_vel_limits=np.ones(7),
        joint_acc_limits=np.ones(7))
    return planner



def follow_path(robot, result,env):
    n_step = result['position'].shape[0]
    for i in range(n_step):  
        qf = robot.compute_passive_force(
            gravity=True, 
            coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        action = result['position'][i]
        print(result['position'][i],action.shape)
        obs, reward, done, info = env.step(action)
        if done:
          env.reset()
        #for j in range(7):
        #    robot.get_joints()[j].set_drive_target(result['position'][i][j])
        #    robot.get_joints()[j].set_drive_velocity_target(result['velocity'][i][j])
        #self.scene.step()
        #if i % 4 == 0:
        #    self.scene.update_render()
        #    self.viewer.render()


def move_to_pose_with_RRT(planner,robot, pose,env):
      result = planner.plan(pose, robot.get_qpos(), time_step=1/250)
      if result['status'] != "Success":
          print(result['status'])
          return -1
      follow_path(robot,result,env)
      return 0

def move_to_pose_with_screw(planner,robot, pose,env):
    result = planner.plan_screw(pose, robot.get_qpos(), time_step=1/250)
    if result['status'] != "Success":
        result = planner.plan(pose, robot.get_qpos(), time_step=1/250)
        if result['status'] != "Success":
            print(result['status'])
            return -1 
    follow_path(robot,result,env)
    return 0

def move_to_pose(pose, with_screw,planner,robot,env):
        if with_screw:
            return move_to_pose_with_screw(planner,robot,pose,env)
        else:
            return move_to_pose_with_RRT(planner,robot,pose,env)



#main
env_id = "LiftCube-v0"
# create environment
env = gym.make(env_id)
# for Colab users we wrap an environment wrapper to auto save videos, no need to learn how RecordEpisode works
env = RecordEpisode(env, "./videos", render_mode="cameras", info_on_video=True)
robot = env.agent.robot
planner = setup_planner(robot)
q = env.obj.get_pose().q
p = env.obj.get_pose().p
pg = env.tcp.get_pose().p
qg = env.tcp.get_pose().q
grip_pose = np.concatenate((pg,qg),axis=0)
obj_pose = np.concatenate((p,q),axis=0)
target_pose = obj_pose + [0,0,0.2,0,0,0,0]
target_grip = grip_pose + [0,1,0,0,0,0,0]
print(env.tcp.get_pose())
print(target_pose)
move_to_pose(target_grip,False,planner,robot,env)
'''
def policy(obs):
    action = np.zeros(env.action_space.shape)
    action[6] = -1
    #print(env.action_space.shape)
    #print(action)
    return action
play(env, policy, steps=100)
'''

# Save the video
env.flush_video()
# close the environment and release resources
env.close()
Video("./videos/0.mp4", embed=True) # Watch our replay