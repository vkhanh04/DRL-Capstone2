import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from buffer import ReplayBuffer
from env import GazeboEnv


def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: avg_reward=%.3f, avg_col=%.3f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward


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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Critic 1
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        # Critic 2
        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        # FIX: Bỏ torch.mm(.weight.data.t()) — cách đó bypass autograd,
        # khiến gradient không chảy qua layer_2_s và layer_2_a đúng cách.
        # Dùng nn.Linear trực tiếp để autograd hoạt động bình thường.

        # Critic 1
        s1 = F.relu(self.layer_1(s))
        s1 = F.relu(self.layer_2_s(s1) + self.layer_2_a(a))
        q1 = self.layer_3(s1)

        # Critic 2
        s2 = F.relu(self.layer_4(s))
        s2 = F.relu(self.layer_5_s(s2) + self.layer_5_a(a))
        q2 = self.layer_6(s2)

        return q1, q2


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=256,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        beta=0.4,  # IS weight cho PER
        beta_max=1.0,  # FIX: Beta annealing — tăng dần đến 1.0 để giảm bias
        total_timesteps=5e6,
        current_timestep=0,
    ):
        av_Q = 0
        max_Q = -float("inf")
        av_loss = 0

        for it in range(iterations):
            # FIX: Beta annealing — tăng tuyến tính từ beta_init → beta_max
            # càng về cuối training, IS weights càng chính xác (bias correction đầy đủ hơn)
            beta_annealed = min(
                beta_max,
                beta + (beta_max - beta) * (current_timestep / total_timesteps),
            )

            # 1. Sample batch từ PER buffer
            batch = replay_buffer.sample_batch(batch_size, beta_annealed)

            state = batch["state"].to(device)
            next_state = batch["next_state"].to(device)
            action = batch["action"].to(device)
            reward = batch["reward"].to(device)
            done = batch["done"].to(device)
            indices = batch["indices"]
            weights = batch["weights"].to(device)

            with torch.no_grad():
                # 2. Target Policy Smoothing
                noise = (torch.randn_like(action) * policy_noise).clamp(
                    -noise_clip, noise_clip
                )
                next_action = (self.actor_target(next_state) + noise).clamp(
                    -self.max_action, self.max_action
                )

                # 3. Clipped Double-Q
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)

                av_Q += torch.mean(target_Q)
                max_Q = max(max_Q, torch.max(target_Q))

                # 4. Bellman target
                target_Q = reward + ((1 - done) * discount * target_Q)

            # 5. Critic forward
            current_Q1, current_Q2 = self.critic(state, action)

            # 6. PER-weighted loss
            td_loss1 = F.mse_loss(current_Q1, target_Q, reduction="none")
            td_loss2 = F.mse_loss(current_Q2, target_Q, reduction="none")
            loss = (weights * (td_loss1 + td_loss2)).mean()

            # 7. Optimize Critic
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            # 8. Cập nhật priorities dựa trên TD-error
            with torch.no_grad():
                new_priorities = (
                    torch.abs(current_Q1 - target_Q).cpu().numpy().flatten()
                )
                replay_buffer.update_priorities(indices, new_priorities)

            # 9. Delayed Actor update
            if it % policy_freq == 0:
                actor_loss, _ = self.critic(state, self.actor(state))
                actor_loss = -actor_loss.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss

        self.iter_count += 1
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


# ── Hyperparameters ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
eval_freq = 5e3  # Sau bao nhiêu steps thì đánh giá
max_ep = 500  # Số bước tối đa mỗi episode
eval_ep = 10  # Số episodes dùng để đánh giá
max_timesteps = 5e6  # Tổng số bước training

expl_noise = 0.3  # Noise khám phá ban đầu
# FIX: Bỏ khai báo expl_min trùng lặp (bản gốc khai báo 2 lần)
expl_min = 0.1  # Noise tối thiểu sau khi decay xong
expl_decay_steps = 500000

# FIX: Tăng batch_size từ 40 → 256 để IS weights ổn định hơn với PER
batch_size = 256
discount = 0.99
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2
buffer_size = 1e6

# PER hyperparameters
beta_init = 0.4  # Beta khởi đầu cho IS annealing
beta_max = 1.0  # Beta cuối (annealing hoàn toàn)

file_name = "TD3_velodyne"
save_model = True
load_model = False
random_near_obstacle = True

# ── Setup ────────────────────────────────────────────────────────────────────
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

network = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(buffer_size, seed)

if load_model:
    try:
        network.load(file_name, "./pytorch_models")
    except Exception:
        print("Could not load stored model, initializing with random parameters")

evaluations = []
timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1
count_rand_actions = 0
random_action = []

# ── Training loop ────────────────────────────────────────────────────────────
while timestep < max_timesteps:

    if done:
        if timestep != 0:
            # FIX: Truyền beta_init và current timestep để annealing hoạt động
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
                beta=beta_init,
                beta_max=beta_max,
                total_timesteps=max_timesteps,
                current_timestep=timestep,
            )

        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            )
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % file_name, evaluations)
            epoch += 1

        state = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Exploration noise decay
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((expl_noise - expl_min) / expl_decay_steps)

    action = network.get_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    # Random action near obstacles để tăng exploration
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[4:-8]) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    replay_buffer.add(state, action, reward, done_bool, next_state)

    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# ── Final evaluation & save ──────────────────────────────────────────────────
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save(file_name, directory="./pytorch_models")
np.save("./results/%s" % file_name, evaluations)
