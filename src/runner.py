import argparse
import random
from typing import Callable
import os
import os.path as osp
import logging

import torch
import numpy as np

from src.model.replay_buffer import ReplayBuffer
from src.utils.normalization import Normalization, RewardScaling
from src.model.opponent_pool import BaseOpponentPool
from env.chooseenv import make


class Runner:
    def __init__(self,
            args: argparse.ArgumentParser,
            num_snakes: int,
            model: torch.nn.Module,
            state_dim: int,
            state_builder: Callable,
            action_dim: int,
            opponent_pool: BaseOpponentPool,
            reward_builder: Callable,
            mask_builder: Callable = None,
            logger: logging.Logger = logging.getLogger(__name__),
        ):
        self.args = args
        self.args.num_snakes = num_snakes
        self.args.state_dim = state_dim
        self.args.action_dim = action_dim
        
        self.num_snakes = num_snakes
        self.state_builder = state_builder
        self.state_dim, self.action_dim = state_dim, action_dim
        self.opponent_pool = opponent_pool
        self.reward_builder = reward_builder
        self.mask_builder = mask_builder
        self.logger = logger
        if getattr(self.opponent_pool, "logger", None) is None:
            self.opponent_pool.logger = self.logger
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        self.env = make("snakes_3v3")  # create environment
        self.replay_buffers = [ReplayBuffer(args) for _ in range(self.num_snakes)]  # create replay buffer
        self.agent = model(args)
        self.logger.info(f"# of params: {sum(p.numel() for p in self.agent.ac.parameters())}")

        self.evaluate_rewards = []  # record the rewards during the evaluating
        self.total_steps = 0

        if self.args.use_state_norm:
            self.logger.info("------use state normalization------")
            self.state_norm = Normalization(shape=self.state_dim, id="state")
        if self.args.use_reward_scaling:
            self.logger.info("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)
        if self.args.load_model:
            self.load_states(self.args.load_dir, self.args.load_n)
        if not osp.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir, exist_ok=True)
        
    def load_states(self, root_dir, n: int = int(3*2e4)):
        """
        Load the model and the running mean and std of the state.

        Args:
            root_dir (str): The root directory of the model.
            n (int): How many steps the model has been trained.
        """

        self.logger.info(f"load model from {osp.abspath(root_dir)}")
        self.agent.load_model(root_dir)

        if self.args.use_state_norm:
            state_stats = np.load(osp.join(root_dir, "state_stats.npz"))
            state_mean, state_std = state_stats["mean"], state_stats["std"]
            self.state_norm.running_ms.n = n
            self.state_norm.running_ms.mean = state_mean
            self.state_norm.running_ms.std = state_std
            self.state_norm.running_ms.S = state_std**2
        
        if self.args.use_reward_scaling:
            reward_stats = np.load(osp.join(root_dir, "reward_stats.npz"))
            reward_mean, reward_std = reward_stats["mean"], reward_stats["std"]
            self.reward_scaling.running_ms.n = n
            self.reward_scaling.running_ms.mean = reward_mean
            self.reward_scaling.running_ms.std = reward_std
            self.reward_scaling.running_ms.S = reward_std**2

    def run(self):
        evaluate_num = 0
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy(self.args.eval_times)
                evaluate_num += 1
            if evaluate_num > 0 and evaluate_num % self.args.save_freq == 0:
                self.agent.save_model(self.args.save_dir)
                if self.args.use_state_norm:
                    self.state_norm.running_ms.save_stats(self.args.save_dir)
                if self.args.use_reward_scaling:
                    self.reward_scaling.running_ms.save_stats(self.args.save_dir)

            episode_steps = self.run_episode()  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffers[0].episode_num * self.num_snakes == self.args.batch_size:
                self.agent.train(self.replay_buffers, self.total_steps)  # Training
                for replay_buffer in self.replay_buffers:
                    replay_buffer.reset_buffer()

        self.evaluate_policy(self.args.eval_times)

    @torch.no_grad()
    def run_episode(self):
        self.agent.ac.eval()
        obs = self.env.reset()
        cur_step = 0
        states = self.state_builder(obs)
        action_masks = self.mask_builder(obs) if self.mask_builder is not None else None
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        self.agent.reset_for_episode()
        actor_hidden_states = [None for _ in range(2*self.num_snakes)]
        critic_hidden_states = [None for _ in range(self.num_snakes)]
        opponent_controller = self.opponent_pool.sample(self.agent, self.state_norm)
        
        for episode_step in range(self.args.episode_max_steps):
            if self.args.use_state_norm:
                for i in range(self.num_snakes):
                    states[i] = self.state_norm(states[i])
            
            # get teammates' actions
            actions, actions_logprob = [], []
            for i in range(self.num_snakes):
                self.agent.ac.actor_rnn_hidden = actor_hidden_states[i]
                action, action_logprob = self.agent.choose_action(states[i], action_masks[i], evaluate=False)
                actor_hidden_states[i] = self.agent.ac.actor_rnn_hidden
                actions.append(action)
                actions_logprob.append(action_logprob)
            
            # get opponents' actions
            opponent_actions = []
            for i in range(self.num_snakes, 2*self.num_snakes):
                opponent_actions.extend(opponent_controller(obs[i], None, None))
            opponent_actions = np.argmax(np.array(opponent_actions), axis=1).tolist()
            actions.extend(opponent_actions)

            # get values
            values = []
            for i in range(self.num_snakes):
                self.agent.ac.critic_rnn_hidden = critic_hidden_states[i]
                values.append(self.agent.get_value(states[i]))
                critic_hidden_states[i] = self.agent.ac.critic_rnn_hidden

            next_obs, rewards, done, _, _ = self.env.step(self.env.encode(actions))
            cur_step += 1
            next_states = self.state_builder(next_obs)
            next_action_masks = self.mask_builder(next_obs) if self.mask_builder is not None else None
            
            dw = True if (done and episode_step + 1 != self.args.episode_max_steps) else False

            # calculate rewards
            rewards = self.reward_builder(next_obs[:self.num_snakes], rewards[:self.num_snakes])
            if self.args.use_reward_scaling:
                for i in range(self.num_snakes):
                    rewards[i] = self.reward_scaling(rewards[i])

            # Store the transition
            for i in range(self.num_snakes):
                self.replay_buffers[i].store_transition(
                    episode_step,
                    states[i],
                    values[i],
                    actions[i],
                    actions_logprob[i],
                    rewards[i],
                    dw,
                    action_masks[i]
                )
            
            obs, states, action_masks = next_obs, next_states, next_action_masks
            
            if done:
                break
        
        self.opponent_pool.update(
            sum([len(obs[0][i+1]) for i in range(self.num_snakes)]) - sum([len(obs[0][i+1]) for i in range(self.num_snakes, 2*self.num_snakes)]),
            eval=False                                                                                       
        )
        
        # An episode is over, store the value in the last step
        if self.args.use_state_norm:
            for i in range(self.num_snakes):
                states[i] = self.state_norm(states[i])
        for i in range(self.num_snakes):
            self.agent.ac.critic_rnn_hidden = critic_hidden_states[i]
            value = self.agent.get_value(states[i])
            critic_hidden_states[i] = self.agent.ac.critic_rnn_hidden
            self.replay_buffers[i].store_last_value(episode_step + 1, value)

        return episode_step + 1

    @torch.no_grad()
    def evaluate_policy(self, eval_times):
        self.agent.ac.eval()
        evaluate_reward = 0
        for _ in range(eval_times):
            scores_period = []
            episode_reward, done = 0, False
            obs = self.env.reset()
            cur_step = 0
            states = self.state_builder(obs)
            masked_actions = self.mask_builder(obs) if self.mask_builder is not None else None
            self.agent.reset_for_episode()
            actor_hidden_states = [None for _ in range(self.num_snakes)]
            while not done:
                if self.args.use_state_norm:
                    for i in range(self.num_snakes):
                        states[i] = self.state_norm(states[i])
                
                actions = []
                for i in range(self.num_snakes):
                    self.agent.ac.actor_rnn_hidden = actor_hidden_states[i]
                    action, _ = self.agent.choose_action(states[i], masked_actions[i], evaluate=True)
                    actor_hidden_states[i] = self.agent.ac.actor_rnn_hidden
                    actions.append(action)

                # compare with the random opponents for fair evaluation
                actions.extend(np.random.randint(low=0, high=self.action_dim-1, size=self.num_snakes).tolist())

                next_obs, r, done, _, _ = self.env.step(self.env.encode(actions))
                
                cur_step += 1
                if cur_step % (self.args.episode_max_steps // 4) == 0:
                    # calculate the score of the current time, refer to the rule of the game (JIDI Snakes3v3).
                    score = len(next_obs[0][2]) + len(next_obs[0][3]) + len(next_obs[0][4]) - 9
                    scores_period.append(score)
                
                next_states = self.state_builder(next_obs)
                next_action_masks = self.mask_builder(next_obs) if self.mask_builder is not None else None
                episode_reward += sum(r[:3])
                obs, states, masked_actions = next_obs, next_states, next_action_masks
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.eval_times
        self.evaluate_rewards.append(evaluate_reward)
        self.logger.info(f"evaluate_num: {int(self.total_steps/self.args.evaluate_freq):>4} / {int(self.args.max_train_steps/self.args.evaluate_freq)} \t evaluate_reward: {evaluate_reward:.3f}")
        
        # update the opponent pool
        self.opponent_pool.update(evaluate_reward, eval=True)
