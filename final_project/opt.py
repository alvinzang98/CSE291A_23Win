
import argparse

def get_opts():

    parser = argparse.ArgumentParser('training')
    #parser.add_argument('-c', '--checkpoint', type=str, default='./model/LiftCube-v1',
    #                    help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--env', default='LiftCube-v1', help='env id')
    parser.add_argument('--traj', default='LiftCube.pkl', type=str, help='data path storing (state,action) pair')
    #parser.add_argument('--policy', type=str, default='VisPolicy', help='policy to use')
    parser.add_argument('--iter', default=10000, type=int, help='iterations in BC')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate of Adam')
    parser.add_argument('--obs', default="rgbd", type=str, help='observation mode')
    parser.add_argument('--control', default="pd_ee_delta_pose", type=str, help='control mode')
    parser.add_argument('--test_eps', default=200, type=int, help='number of episode used during evaluation')

    return parser.parse_args()