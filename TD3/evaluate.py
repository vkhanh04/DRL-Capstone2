"""
evaluate.py
-----------
Script đánh giá success rate của model TD3 trong môi trường Gazebo.
Chạy N episodes, ghi lại kết quả chi tiết và lưu ra file .csv + .txt

Cách dùng:
    python3 evaluate.py
    python3 evaluate.py --episodes 100 --model TD3_velodyne --max_steps 500
"""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import GazeboEnv


# ── Network (giữ nguyên kiến trúc để load đúng weights) ──────────────────────
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
        return self.tanh(self.layer_3(s))


class TD3(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load(
                "%s/%s_actor.pth" % (directory, filename),
                map_location=device,
                weights_only=True,  # tắt FutureWarning của PyTorch 2.x
            )
        )


# ── Evaluation logic ──────────────────────────────────────────────────────────
def run_evaluation(network, env, n_episodes, max_steps):
    """
    Chạy n_episodes episodes và trả về dict kết quả chi tiết.

    Phân loại kết quả mỗi episode:
      - SUCCESS  : robot đến goal
      - COLLISION: robot va chạm obstacle
      - TIMEOUT  : hết max_steps mà chưa đến goal và chưa collision
    """
    results = []

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        step = 0
        outcome = "TIMEOUT"  # default

        while not done and step < max_steps:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            next_state, reward, done, target = env.step(a_in)

            episode_reward += reward
            step += 1

            if done:
                if target:
                    outcome = "SUCCESS"
                else:
                    outcome = "COLLISION"

            state = next_state

        results.append(
            {
                "episode": ep,
                "outcome": outcome,
                "steps": step,
                "reward": round(episode_reward, 3),
            }
        )

        # In tiến trình real-time
        icon = (
            "✅" if outcome == "SUCCESS" else ("💥" if outcome == "COLLISION" else "⏱️")
        )
        print(
            f"  Ep {ep:>3}/{n_episodes} | {icon} {outcome:<10} | "
            f"Steps: {step:>3} | Reward: {episode_reward:>8.2f}"
        )

    return results


def compute_stats(results):
    """Tính các thống kê từ danh sách kết quả."""
    n = len(results)
    n_success = sum(1 for r in results if r["outcome"] == "SUCCESS")
    n_collision = sum(1 for r in results if r["outcome"] == "COLLISION")
    n_timeout = sum(1 for r in results if r["outcome"] == "TIMEOUT")

    rewards = [r["reward"] for r in results]
    steps_success = [r["steps"] for r in results if r["outcome"] == "SUCCESS"]

    stats = {
        "total_episodes": n,
        "success_count": n_success,
        "collision_count": n_collision,
        "timeout_count": n_timeout,
        "success_rate": round(n_success / n * 100, 2),
        "collision_rate": round(n_collision / n * 100, 2),
        "timeout_rate": round(n_timeout / n * 100, 2),
        "avg_reward": round(np.mean(rewards), 3),
        "std_reward": round(np.std(rewards), 3),
        "min_reward": round(np.min(rewards), 3),
        "max_reward": round(np.max(rewards), 3),
        "avg_steps_on_success": (
            round(np.mean(steps_success), 1) if steps_success else 0
        ),
    }
    return stats


def print_report(stats, model_name, timestamp):
    """In báo cáo đẹp ra terminal."""
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  EVALUATION REPORT")
    print(f"  Model   : {model_name}")
    print(f"  Time    : {timestamp}")
    print(sep)
    print(f"  Total episodes : {stats['total_episodes']}")
    print(f"")
    print(
        f"  ✅ SUCCESS   : {stats['success_count']:>3} episodes  "
        f"→  {stats['success_rate']:>6.2f}%"
    )
    print(
        f"  💥 COLLISION : {stats['collision_count']:>3} episodes  "
        f"→  {stats['collision_rate']:>6.2f}%"
    )
    print(
        f"  ⏱️  TIMEOUT   : {stats['timeout_count']:>3} episodes  "
        f"→  {stats['timeout_rate']:>6.2f}%"
    )
    print(f"")
    print(
        f"  Avg reward        : {stats['avg_reward']:>8.3f}  "
        f"(± {stats['std_reward']:.3f})"
    )
    print(
        f"  Min / Max reward  : {stats['min_reward']:>8.3f}  "
        f"/ {stats['max_reward']:.3f}"
    )
    print(f"  Avg steps (success): {stats['avg_steps_on_success']:>7.1f} steps")
    print(sep)


def save_results(results, stats, model_name, output_dir, timestamp):
    """Lưu kết quả ra CSV và TXT để dùng cho báo cáo."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"{model_name}_{timestamp}"

    # 1. CSV — chi tiết từng episode (dùng để vẽ biểu đồ)
    csv_path = os.path.join(output_dir, f"{base_name}_episodes.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "outcome", "steps", "reward"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  📄 Episode details saved → {csv_path}")

    # 2. TXT — summary stats (copy vào báo cáo)
    txt_path = os.path.join(output_dir, f"{base_name}_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total episodes: {stats['total_episodes']}\n\n")
        f.write(
            f"SUCCESS rate  : {stats['success_rate']}%  ({stats['success_count']} eps)\n"
        )
        f.write(
            f"COLLISION rate: {stats['collision_rate']}%  ({stats['collision_count']} eps)\n"
        )
        f.write(
            f"TIMEOUT rate  : {stats['timeout_rate']}%  ({stats['timeout_count']} eps)\n\n"
        )
        f.write(f"Avg reward : {stats['avg_reward']} ± {stats['std_reward']}\n")
        f.write(f"Min reward : {stats['min_reward']}\n")
        f.write(f"Max reward : {stats['max_reward']}\n")
        f.write(f"Avg steps (success only): {stats['avg_steps_on_success']}\n")
    print(f"  📄 Summary saved        → {txt_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TD3 model success rate")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Số episodes đánh giá (default: 100)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Số bước tối đa mỗi episode (default: 500)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TD3_velodyne",
        help="Tên file model (default: TD3_velodyne)",
    )
    parser.add_argument(
        "--model_dir", type=str, default="./pytorch_models", help="Thư mục chứa model"
    )
    parser.add_argument(
        "--output", type=str, default="./eval_results", help="Thư mục lưu kết quả"
    )
    parser.add_argument(
        "--env_dim",
        type=int,
        default=36,
        help="environment_dim — phải khớp với lúc train (default: 36)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup
    print(f"\n{'='*55}")
    print(f"  TD3 Evaluation Script")
    print(f"  Device  : {device}")
    print(f"  Model   : {args.model}")
    print(f"  Episodes: {args.episodes}  |  Max steps: {args.max_steps}")
    print(f"{'='*55}\n")

    robot_dim = 4
    state_dim = args.env_dim + robot_dim
    action_dim = 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load environment
    env = GazeboEnv("multi_robot_scenario.launch", args.env_dim)
    time.sleep(5)

    # Load model
    network = TD3(state_dim, action_dim)
    try:
        network.load(args.model, args.model_dir)
        print(f"  ✅ Model loaded: {args.model_dir}/{args.model}_actor.pth\n")
    except Exception as e:
        raise ValueError(f"Cannot load model: {e}")

    # Run evaluation
    print(f"  Running {args.episodes} evaluation episodes...\n")
    results = run_evaluation(network, env, args.episodes, args.max_steps)

    # Compute & display stats
    stats = compute_stats(results)
    print_report(stats, args.model, timestamp)

    # Save
    save_results(results, stats, args.model, args.output, timestamp)
    print(f"\n  Done.\n")
