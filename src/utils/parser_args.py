import argparse


def parse_args():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--eval_times", type=int, default=2, help="Evaluate times")
    parser.add_argument("--episode_max_steps", type=int, default=200, help="Maximum steps in one episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--load_model", action="store_true", default=False, help="Whether to load model")
    parser.add_argument("--load_dir", type=str, default=None, help="The path of model loading")
    parser.add_argument("--load_n", type=int, default=int(3*2e4), help="How many steps the model has been trained")
    parser.add_argument("--save_dir", type=str, default="./model", help="The path of model saving")

    parser.add_argument("--batch_size", type=int, default=60, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=30, help="Minibatch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--hidden_state_dim", type=int, default=64, help="The number of hidden state neurons in LSTM")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="PPO parameter")
    parser.add_argument("--use_adv_norm", action="store_true", default=True, help="Whether to use advantage normalization")
    parser.add_argument("--use_state_norm", action="store_true", default=True, help="Whether to use state normalization")
    parser.add_argument("--use_reward_scaling", action="store_true", default=True, help="Whether to use reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Whether to use policy entropy")
    parser.add_argument("--use_lr_decay", action="store_true", default=True, help="Whether to use learning rate Decay")
    parser.add_argument("--use_grad_clip", action="store_true", default=True, help="Whether to use Gradient clip")
    parser.add_argument("--use_orthogonal_init", action="store_true", default=True, help="Whether to use orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Whether to set Adam epsilon to 1e-5")

    args = parser.parse_args()

    return args