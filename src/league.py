save_league_freq = 1000

if __name__ == "__main__":
    from utils.parser_args import parse_args
    args = parse_args()
    
    import os.path as osp
    from config.config import get_config
    config = get_config("league",
                        args=args,
                        league_dir=osp.join(args.save_dir, "league"),
                        save_league_freq=save_league_freq)
    
    from src.runner import Runner
    runner = Runner(args, **config)
    runner.run()