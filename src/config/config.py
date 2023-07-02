def get_config(mode: str = "main", **kwargs) -> dict:
    if mode == "main":
        from src.model.ppo_discrete_rnn import PPODiscreteRNN
        from src.model.opponent_pool import OpponentPool
        from src.model.builders import state_builder, reward_builder, mask_builder
        from src.utils.logger import get_logger

        config = {
            "num_snakes": 3,
            "model": PPODiscreteRNN,
            "state_dim": 17,
            "state_builder": state_builder,
            "action_dim": 4,
            "opponent_pool": OpponentPool(reward_threshold=30),
            "reward_builder": reward_builder,
            "mask_builder": mask_builder,
            "logger": get_logger("./logs/log_main.txt"),
        }
        return config
    elif mode == "league":
        from src.model.ppo_discrete_rnn import PPODiscreteRNN
        from src.model.opponent_pool import League
        from src.model.builders import state_builder, reward_builder, mask_builder
        from src.utils.logger import get_logger

        num_snakes = 3
        state_dim = 17
        action_dim = 4
        
        args = kwargs["args"]
        args.num_snakes = num_snakes
        args.state_dim = state_dim
        args.action_dim = action_dim
        config = {
            "num_snakes": num_snakes,
            "model": PPODiscreteRNN,
            "state_dim": state_dim,
            "state_builder": state_builder,
            "action_dim": action_dim,
            "opponent_pool": League(args, kwargs["league_dir"], kwargs["save_league_freq"]),
            "reward_builder": reward_builder,
            "mask_builder": mask_builder,
            "logger": get_logger("./logs/log_league.txt"),
        }
        return config
    else:
        raise ValueError("Invalid mode")