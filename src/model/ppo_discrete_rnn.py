import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer


class ACRNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.activate_func = nn.ReLU()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        
        # actor
        self.actor_rnn_hidden = None
        self.actor_fc1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.actor_rnn = nn.LSTM(args.hidden_dim, args.hidden_state_dim, batch_first=True)
        self.actor_fc2 = nn.Linear(args.hidden_state_dim, args.action_dim)

        self.critic_rnn_hidden = None
        self.critic_fc1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.critic_rnn = nn.LSTM(args.hidden_dim, args.hidden_state_dim, batch_first=True)
        self.critic_fc2 = nn.Linear(args.hidden_state_dim, 1)
        
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.actor_fc1)
            orthogonal_init(self.actor_rnn)
            orthogonal_init(self.actor_fc2, gain=0.01)
            orthogonal_init(self.critic_fc1)
            orthogonal_init(self.critic_rnn)
            orthogonal_init(self.critic_fc2)

    def common_forward(self, state):
        out = self.activate_func(self.fc1(state))
        return out
    
    def actor(self, state):
        embedding = self.common_forward(state)
        embedding = self.activate_func(self.actor_fc1(embedding))
        self.actor_rnn.flatten_parameters()
        output, self.actor_rnn_hidden = self.actor_rnn(embedding, self.actor_rnn_hidden)
        logit = self.actor_fc2(output)
        return logit

    def critic(self, state):
        embedding = self.common_forward(state)
        embedding = self.activate_func(self.critic_fc1(embedding))
        self.critic_rnn.flatten_parameters()
        output, self.critic_rnn_hidden = self.critic_rnn(embedding, self.critic_rnn_hidden)
        value = self.critic_fc2(output)
        return value


class PPODiscreteRNN:
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr  # Learning rate of actor
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.ac = ACRNN(args).to(DEVICE)
        if self.set_adam_eps:
            self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.lr, eps=1e-5)
        else:
            self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.lr)

    def reset_for_episode(self):
        self.ac.actor_rnn_hidden = None
        self.ac.critic_rnn_hidden = None

    @torch.no_grad()
    def choose_action(self, state, action_mask, evaluate=False):
        logit = self.ac.actor(state.unsqueeze(0).to(DEVICE)).squeeze(0)
        if evaluate:
            action = torch.argmax(logit)
            return action.item(), None
        else:
            logit[action_mask == 1] = -1e7
            dist = Categorical(logits=logit)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            return action.item(), action_logprob.item()

    @torch.no_grad()
    def get_value(self, state):
        value = self.ac.critic(state.unsqueeze(0).to(DEVICE))
        return value.item()

    def train(self, replay_buffers, total_steps):
        self.ac.train()
        batch_sep = []
        for i in range(self.args.num_snakes):
            batch_sep.append(replay_buffers[i].get_training_data())
        batch = {attr:torch.cat((batch_sep[0][attr], batch_sep[1][attr], batch_sep[2][attr]), dim=0) for attr in batch_sep[0].keys()}

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                self.reset_for_episode()
                logits = self.ac.actor(batch["state"][index].to(DEVICE))  # logits.shape=(mini_batch_size, max_episode_len, action_dim)
                logits[batch["action_mask"][index] == 1] = -1e7
                values = self.ac.critic(batch["state"][index].to(DEVICE)).squeeze(-1)  # values_now.shape=(mini_batch_size, max_episode_len)

                dist = Categorical(logits=logits.cpu())
                dist_entropy = dist.entropy()  # shape(mini_batch_size, max_episode_len)
                a_logprob = dist.log_prob(batch["action"][index])  # shape(mini_batch_size, max_episode_len)
                ratios = torch.exp(a_logprob - batch["action_logprob"][index])  # shape(mini_batch_size, max_episode_len)

                # actor loss
                surr1 = ratios * batch["adv"][index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch["adv"][index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size, max_episode_len)
                actor_loss = (actor_loss * batch["active"][index]).sum() / batch["active"][index].sum()

                # critic_loss
                critic_loss = (values - batch["target_value"][index].to(DEVICE)) ** 2
                critic_loss = (critic_loss * batch["active"][index].to(DEVICE)).sum() / batch["active"][index].sum().to(DEVICE)

                # Update
                self.optimizer.zero_grad()
                loss = actor_loss + critic_loss * 0.5
                loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p["lr"] = lr_now

    def save_model(self, save_dir):
        save_ac_path = osp.join(save_dir, "ac.pth")
        torch.save(self.ac.state_dict(), save_ac_path)
    
    def load_model(self, save_dir):
        save_ac_path = osp.join(save_dir, "ac.pth")
        self.ac.load_state_dict(torch.load(save_ac_path, map_location=DEVICE))
