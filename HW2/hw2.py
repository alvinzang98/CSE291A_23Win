import gym
from tqdm.notebook import tqdm,trange
import numpy as np
import mani_skill2.envs
import matplotlib.pyplot as plt
import sapien.core as sapien
import mplib
import numpy as np
from sapien.utils.viewer import Viewer
from mani_skill2.utils.wrappers import RecordEpisode
from IPython.display import Video


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
    env.reset()
    for i in trange(n_step):  
        qf = robot.compute_passive_force(
            gravity=True, 
            coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        action = result['position'][i]
        action = np.concatenate((action,[0]))
        #print(action)      #len:9
        obs, reward, done, info = env.step(action)
        if done:
          env.reset()
        


def move_to_pose_with_RRT(planner,robot, pose,env):
      
      result = planner.plan(pose, robot.get_qpos(), time_step=1/250)
      
      if result['status'] != "Success":
         
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


def open_gripper(self,robot,env):
    for joint in robot.get_joints()[-2:]:
        joint.set_drive_target(0.4)
    for i in range(100): 
        qf = robot.compute_passive_force(
            gravity=True, 
            coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        action = robot.get_qpos()[:7]
        action = np.concatenate((action,[0.4]))
        env.step(action)
            

def close_gripper(self,robot,env):
    for joint in robot.get_joints()[-2:]:
        joint.set_drive_target(0)
    for i in range(100):  
        qf = robot.compute_passive_force(
            gravity=True, 
            coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        action = robot.get_qpos()[:7]
        action = np.concatenate((action,[0]))
        env.step(action)
        


def run(self,obj_pose, with_screw,planner,robot,env):
        
    obj_pose[2] += 0.2
    move_to_pose(obj_pose,with_screw,planner,robot,env)
    open_gripper(robot,env)
    obj_pose[2] -= 0.16
    move_to_pose(obj_pose,with_screw,planner,robot,env)
    close_gripper(robot,env)
    obj_pose[2] += 0.16
    move_to_pose(obj_pose,with_screw,planner,robot,env)
    

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
grip_pose = np.concatenate((pg,qg),axis=0).tolist()
obj_pose = np.concatenate((p,q),axis=0).tolist()
obj_pose[2] += 0.02

run(obj_pose,False,planner,robot,env)



# Save the video
env.flush_video()
# close the environment and release resources
env.close()
Video("./videos/0.mp4", embed=True) # Watch our replay


