import pickle
from torch.utils.data import Dataset, DataLoader
import torch as th
import numpy as np
from tqdm import tqdm

def tensor_to_numpy(x):
    # moves all tensors to numpy. This is just for SB3 as SB3 does not optimize for observations stored on the GPU.
    if th.is_tensor(x):
        return x.cpu().numpy()
    return x
def convert_observation(observation):
    # flattens the original observation by flattening the state dictionaries
    # and combining the rgb and depth images
    # image data is not scaled here and is kept as uint16 to save space
    image_obs = observation["image"]
    rgb = image_obs["base_camera"]["rgb"]
    depth = image_obs["base_camera"]["depth"]
    rgb2 = image_obs["hand_camera"]["rgb"]
    depth2 = image_obs["hand_camera"]["depth"]

    # we provide a simple tool to flatten dictionaries with state data
    from mani_skill2.utils.common import flatten_state_dict
    state = np.hstack(
        [
            flatten_state_dict(observation["agent"]),
            flatten_state_dict(observation["extra"]),
        ]
    )
    # combine the RGB and depth images
    rgbd = np.concatenate([rgb, depth, rgb2, depth2], axis=-1)
    obs = dict(rgbd=rgbd, state=state)
    return obs
def rescale_rgbd(rgbd):
    # rescales rgbd data and changes them to floats
    rgb1 = rgbd[..., 0:3] / 255.0
    rgb2 = rgbd[..., 4:7] / 255.0
    depth1 = rgbd[..., 3:4] / (2**10)
    depth2 = rgbd[..., 7:8] / (2**10) 
    return np.concatenate([rgb1, depth1, rgb2, depth2], axis=-1)
class ManiSkill2Dataset(Dataset):
    def __init__(self, dataset_file: str, load_count=-1) -> None:
        self.dataset_file = dataset_file
        import pickle
        self.data = pickle.load(open('./data.pkl', 'rb'))
        self.episodes = len(self.data.keys())

        self.obs_state = []
        self.obs_rgbd = []
        self.actions = []
        self.total_frames = 0
        if load_count == -1:
            load_count = self.episodes
        for eps_id in tqdm(range(load_count)):
            # eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps_id}"]

            # convert the original raw observation with our batch-aware function
            obs = convert_observation(trajectory["obs"])
            # we use :-1 to ignore the last obs as terminal observations are included
            # and they don't have actions
            self.obs_rgbd.append(obs['rgbd'][:-1])
            self.obs_state.append(obs['state'][:-1])
            self.actions.append(trajectory["actions"])
        self.obs_rgbd = np.vstack(self.obs_rgbd)
        self.obs_state = np.vstack(self.obs_state)
        self.actions = np.vstack(self.actions)

    def __len__(self):
        return len(self.obs_rgbd)

    def __getitem__(self, idx):
        action = th.from_numpy(self.actions[idx]).float()
        rgbd = self.obs_rgbd[idx]
        rgbd = rescale_rgbd(rgbd)
        # permute data so that channels are the first dimension as PyTorch expects this
        rgbd = th.from_numpy(rgbd).float().permute((2, 0, 1))
        state = th.from_numpy(self.obs_state[idx]).float()
        return dict(rgbd=rgbd, state=state), action
    
if __name__=="__main__":
    dataset = ManiSkill2Dataset(f"data.pkl")
    dataloader = DataLoader(dataset, batch_size=100, num_workers=1, pin_memory=True, drop_last=True, shuffle=True)
    obs, action = dataset[0]
    print("RGBD:", obs['rgbd'].shape)
    print("State:", obs['state'].shape)
    print("Action:", action.shape)