import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size=1000000, seed=0, alpha=0.6):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.alpha = alpha  # Độ quyết định mức độ prioritized (0: ngẫu nhiên, 1: prioritized hoàn toàn)

        # FIX: Khởi tạo random state riêng để không ảnh hưởng global seed
        self.rng = np.random.default_rng(seed)

        self.state_dim = 24
        self.action_dim = 2

        self.S = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.NS = np.zeros_like(self.S)
        self.A = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.R = np.zeros(self.max_size, dtype=np.float32)
        self.D = np.zeros(self.max_size, dtype=np.float32)

        # Lưu trữ độ ưu tiên cho mỗi sample
        self.priorities = np.zeros(self.max_size, dtype=np.float32)

    def add(self, s, a, r, d, ns):
        self.S[self.ptr] = s
        self.A[self.ptr] = a
        self.R[self.ptr] = r
        self.NS[self.ptr] = ns
        self.D[self.ptr] = d

        # Gán độ ưu tiên cao nhất cho sample mới để đảm bảo nó được học ít nhất một lần
        max_priority = self.priorities[: self.size].max() if self.size > 0 else 1.0
        self.priorities[self.ptr] = max(max_priority, 1.0)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=64, beta=0.4):
        # Tính xác suất lấy mẫu dựa trên priorities
        priorities_slice = self.priorities[: self.size]
        probs = priorities_slice**self.alpha
        probs /= probs.sum()

        # FIX: Dùng self.rng thay vì np.random.choice để tránh ảnh hưởng global seed
        indices = self.rng.choice(self.size, batch_size, p=probs, replace=True)

        # Tính Importance Sampling weights để bù đắp việc lấy mẫu lệch (bias)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize để ổn định training
        weights = torch.FloatTensor(weights).unsqueeze(1)

        batch = {
            "state": torch.FloatTensor(self.S[indices]),
            "next_state": torch.FloatTensor(self.NS[indices]),
            "action": torch.FloatTensor(self.A[indices]),
            "reward": torch.FloatTensor(self.R[indices]).unsqueeze(1),
            "done": torch.FloatTensor(self.D[indices]).unsqueeze(1),
            "indices": indices,
            "weights": weights,
        }
        return batch

    def update_priorities(self, indices, errors):
        # FIX: Vectorized thay vì Python loop - nhanh hơn đáng kể với batch lớn
        # Cập nhật priorities dựa trên TD-error mới từ Critic
        self.priorities[indices] = (
            np.abs(errors) + 1e-6
        )  # Thêm epsilon để tránh priority = 0
