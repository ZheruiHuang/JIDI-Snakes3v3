import copy
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.use_adv_norm = args.use_adv_norm
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.episode_limit = args.episode_max_steps
        assert args.batch_size % args.num_snakes == 0
        self.batch_size = args.batch_size // args.num_snakes
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {
            "state": np.zeros((self.batch_size, self.episode_limit) + (self.state_dim,)),
            "value": np.zeros([self.batch_size, self.episode_limit + 1]),
            "action": np.zeros([self.batch_size, self.episode_limit]),
            "action_logprob": np.zeros([self.batch_size, self.episode_limit]),
            "reward": np.zeros([self.batch_size, self.episode_limit]),
            "dw": np.ones([self.batch_size, self.episode_limit]),  # Note: We use 'np.ones' to initialize 'dw'
            "action_mask": np.zeros([self.batch_size, self.episode_limit, self.action_dim]),
            "active": np.zeros([self.batch_size, self.episode_limit])
        }
        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, state, value, action, action_logprob, reward, dw, action_mask):
        self.buffer["state"][self.episode_num][episode_step] = state
        self.buffer["value"][self.episode_num][episode_step] = value
        self.buffer["action"][self.episode_num][episode_step] = action
        self.buffer["action_logprob"][self.episode_num][episode_step] = action_logprob
        self.buffer["reward"][self.episode_num][episode_step] = reward
        self.buffer["dw"][self.episode_num][episode_step] = dw
        self.buffer["action_mask"][self.episode_num][episode_step] = action_mask
        self.buffer["active"][self.episode_num][episode_step] = 1.0

    def store_last_value(self, episode_step, value):
        self.buffer["value"][self.episode_num][episode_step] = value
        self.episode_num += 1
        # Record max_episode_len
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step

    @torch.no_grad()
    def get_adv(self):
        # Calculate the advantage using GAE
        v = self.buffer["value"][:, :self.max_episode_len]
        v_next = self.buffer["value"][:, 1:self.max_episode_len + 1]
        r = self.buffer["reward"][:, :self.max_episode_len]
        dw = self.buffer["dw"][:, :self.max_episode_len]
        active = self.buffer["active"][:, :self.max_episode_len]
        adv = np.zeros_like(r)  # adv.shape=(batch_size,max_episode_len)
        gae = 0
        # deltas.shape=(batch_size,max_episode_len)
        deltas = r + self.gamma * v_next * (1 - dw) - v
        for t in reversed(range(self.max_episode_len)):
            gae = deltas[:, t] + self.gamma * self.lamda * gae  # gae.shape=(batch_size)
            adv[:, t] = gae
        v_target = adv + v  # v_target.shape(batch_size,max_episode_len)
        if self.use_adv_norm:
            adv_copy = copy.deepcopy(adv)
            adv_copy[active == 0] = np.nan  # ignore the inactive steps
            adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))
        return adv, v_target

    def get_training_data(self):
        adv, v_target = self.get_adv()
        batch = {
            "state": torch.tensor(self.buffer["state"][:, :self.max_episode_len], dtype=torch.float32),
            "action": torch.tensor(self.buffer["action"][:, :self.max_episode_len], dtype=torch.long),
            "action_logprob": torch.tensor(self.buffer["action_logprob"][:, :self.max_episode_len], dtype=torch.float32),
            "active": torch.tensor(self.buffer["active"][:, :self.max_episode_len], dtype=torch.float32),
            "adv": torch.tensor(adv, dtype=torch.float32),
            "target_value": torch.tensor(v_target, dtype=torch.float32),
            "action_mask": torch.tensor(self.buffer["action_mask"][:, :self.max_episode_len], dtype=torch.float32)
        }
        return batch
