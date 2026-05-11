import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import GazeboEnv


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class TD3(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        # FIX: Thêm map_location để load đúng thiết bị (tránh lỗi khi train GPU, test CPU)
        self.actor.load_state_dict(
            torch.load(
                "%s/%s_actor.pth" % (directory, filename),
                map_location=device,
            )
        )


# ── Parameters ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
max_ep = 500
file_name = "TD3_velodyne"

# ── Setup ─────────────────────────────────────────────────────────────────────
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = environment_dim + robot_dim
action_dim = 2

network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./pytorch_models")
    print("Model loaded successfully from ./pytorch_models/%s_actor.pth" % file_name)
except Exception as e:
    raise ValueError("Could not load the stored model parameters: %s" % str(e))

# ── Testing loop ──────────────────────────────────────────────────────────────
done = False
episode_timesteps = 0
episode_num = 0
episode_reward = 0.0
state = env.reset()

print("Starting test loop...")

while True:
    action = network.get_action(np.array(state))

    # Map action: linear velocity → [0, 1], angular velocity → [-1, 1]
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    episode_reward += reward

    # FIX: Tách biến done_flag khỏi done để tránh nhầm lẫn
    timeout = episode_timesteps + 1 == max_ep
    done_flag = timeout or bool(done)

    if done_flag:
        episode_num += 1
        goal_str = "GOAL REACHED" if target else ("TIMEOUT" if timeout else "COLLISION")
        print(
            "Episode %d | Steps: %d | Reward: %.2f | Result: %s"
            % (episode_num, episode_timesteps + 1, episode_reward, goal_str)
        )

        # Reset
        state = env.reset()
        done = False
        episode_timesteps = 0
        episode_reward = 0.0
    else:
        state = next_state
        episode_timesteps += 1
