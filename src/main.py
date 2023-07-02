if __name__ == "__main__":
    from utils.parser_args import parse_args
    args = parse_args()
    
    from config.config import get_config
    config = get_config("main")
    
    from src.runner import Runner
    runner = Runner(args, **config)
    runner.run()