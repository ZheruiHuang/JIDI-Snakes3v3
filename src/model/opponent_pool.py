import os
import random
from typing import Callable
import torch


class BaseOpponentPool:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def sample(self, *args, **kwargs) -> Callable:
        raise NotImplementedError
    
    def update(self, reward: int, eval: bool, *args, **kwargs) -> None:
        raise NotImplementedError


class OpponentPool(BaseOpponentPool):
    def __init__(self, reward_threshold=30, logger=None):
        super().__init__()
        # import controllers
        from agents.always_up.submission import my_controller as always_up_controller
        from agents.random.submission import my_controller as random_controller
        # from agents.maddpg.submission import my_controller as maddpg_controller
        # from agents.ppo_norule.submission import my_controller as ppo_norule_controller
        # from agents.ppo_rule_enhanced.submission import my_controller as ppo_rule_controller
        # from agents.ppo_rnn.submission import my_controller as ppo_rnn_controller
        
        self.logger = logger
        
        # initialize opponent pool
        self.opponent_pool = []
        self.opponent_pool.extend([0] * 2)  # always up
        self.opponent_pool.extend([1] * 3)  # random
        # self.opponent_pool.extend([2] * 20)  # maddpg
        # self.opponent_pool.extend([3] * 20)  # no-rule ppo
        # self.opponent_pool.extend([4] * 20)  # rule-enhanced ppo
        # self.opponent_pool.extend([5] * 35)  # ppo_rnn
        self.num_opponent_dict = {
            0: always_up_controller,
            1: random_controller,
            # 2: maddpg_controller,
            # 3: ppo_norule_controller,
            # 4: ppo_rule_controller,
            # 5: ppo_rnn_controller
        }
        self.num_opponent = len(self.opponent_pool)
        
        self.num_great_reward = 0
        self.reward_threshold = reward_threshold
        
    def sample(self, *args, **kwargs):
        return self.num_opponent_dict[random.choice(self.opponent_pool[:self.num_great_reward+1])]
    
    def update(self, reward, eval, *args, **kwargs):
        if not eval: return
        if self.num_great_reward < self.num_opponent-1 \
            and reward >= self.reward_threshold:
            self.num_great_reward += 1
            info = f"Reached reward threshold {self.reward_threshold} for {self.num_great_reward} times"
            if self.logger is not None:
                self.logger.info(info)
            else:
                print(info)


class League(BaseOpponentPool):
    def __init__(self, args, league_dir: str, save_league_freq: int, logger=None):
        super().__init__()
        from src.model.ppo_discrete_rnn import PPODiscreteRNN
        from src.utils.normalization import Normalization

        self.leagur_dir = league_dir
        if not os.path.exists(league_dir):
            os.makedirs(league_dir, exist_ok=True)
        self.save_league_freq = save_league_freq
        self.logger = logger
        self.opponent_pool = []
        self.opponent_pool.extend([0] * 4)  # self-play
        self.opponent_pool.extend([1] * 6)  # vs. historical agents
        self.opponent_lossingRate_dict = {}
        self.sample_num, self.opponent_num = 0, 0
        self.args = args
        self.opponent_idx = -1
        self.opponent = PPODiscreteRNN(self.args)
        self.state_norm = Normalization(shape=args.state_dim, id="state")
    
    def sample(self, _self_model, _self_state_norm, *args, **kwargs):
        if self.opponent_num == 0 or random.choice(self.opponent_pool) == 0:
            model, state_norm = _self_model, _self_state_norm
            self.opponent_idx = -1
        else:
            model, state_norm = self.load_opponent()
        
        self.sample_num += 1
        if self.sample_num // self.save_league_freq > self.opponent_num:
            save_model_dir = os.path.join(self.leagur_dir, f"agent_{self.opponent_num}")
            if not os.path.exists(save_model_dir):
                os.makedirs(save_model_dir, exist_ok=True)
            _self_model.save_model(save_model_dir)
            _self_state_norm.running_ms.save_stats(save_model_dir)
            self.opponent_lossingRate_dict[self.opponent_num] = [1, 0]
            self.opponent_num += 1
            
            from tabulate import tabulate
            table = zip(
                self.opponent_lossingRate_dict.keys(),
                [1.-(x[0]/x[1] if x[1]>0 else 1.) for x in self.opponent_lossingRate_dict.values()],
                self.opponent_lossingRate_dict.values()
            )
            info = f"Save the current agent to League.\n{tabulate(table, headers=['vs. agent', 'winning rate', 'loss&draw / total'], tablefmt='grid')}"
            if self.logger is not None:
                self.logger.info(info)
            else:
                print(info)
        
        return self.build_opponent(model, state_norm)
    
    def update(self, reward, eval, *args, **kwargs):
        if eval: return
        if self.opponent_idx >= 0:
            self.opponent_lossingRate_dict[self.opponent_idx][1] += 1
            if reward <= 0:
                self.opponent_lossingRate_dict[self.opponent_idx][0] += 1
    
    def load_opponent(self):
        opponents = list(self.opponent_lossingRate_dict.keys())
        opponents_lossingRate = \
            [(x[0]/x[1] if x[1]>0 else 0.5) for x in self.opponent_lossingRate_dict.values()]
        opponent = random.choices(opponents, opponents_lossingRate)[0]
        self.opponent_idx = opponent
        opponent_model_dir = os.path.join(self.leagur_dir, f"agent_{opponent}")
        self.opponent.load_model(opponent_model_dir)
        self.opponent.ac.eval()

        import numpy as np
        if self.args.use_state_norm:
            state_stats = np.load(os.path.join(opponent_model_dir, "state_stats.npz"))
            state_mean, state_std = state_stats["mean"], state_stats["std"]
            self.state_norm.running_ms.n = 1
            self.state_norm.running_ms.mean = state_mean
            self.state_norm.running_ms.std = state_std
            self.state_norm.running_ms.S = state_std**2
        
        return self.opponent, self.state_norm
        
    def build_opponent(self, model, state_norm) -> Callable:
        return self.Opponent(self.args, model, state_norm)

    class Opponent:
        from src.model.ppo_discrete_rnn import PPODiscreteRNN
        from src.utils.normalization import Normalization
        def __init__(self, args, model: PPODiscreteRNN, state_norm: Normalization):
            self.model = model
            self.model.ac.eval()
            self.state_norm = state_norm
            self.action_dim = args.action_dim
            self.actor_hidden_states = [None for _ in range(2*args.num_snakes)]
            self.action_masks = [None for _ in range(2*args.num_snakes)]

        def __call__(self, ob, *args, **kwargs):
            from src.model.builders import state_builder, mask_builder
            self_snake_id = ob["controlled_snake_index"] - 2
            state = state_builder([ob])[0]
            state = self.state_norm(state, update=False).to(torch.float)
            action_mask = mask_builder([ob])[0]
            self.model.ac.actor_rnn_hidden = self.actor_hidden_states[self_snake_id]
            taken_action, _ = self.model.choose_action(state, action_mask, evaluate=True)
            self.actor_hidden_states[self_snake_id] = self.model.ac.actor_rnn_hidden
            action = [0 for _ in range(self.action_dim)]
            action[taken_action] = 1
            return [action]