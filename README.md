# JIDI-Snakes3v3
Project for SJTU AI3601.

## Description

Using technique-enhancing MAPPO algorithm for specific multi-agent reinforcement learning task Snakes 3v3.
For this specific task, we design features, reshape rewards and use several techniques to improve the agent's performance. The techniques include but are not limited to:

- [x] Normalization
- [x] Policy Entropy
- [x] Learning Rate Decay
- [x] Curriculum Learning
- [x] Self-play Enhancement
- [x] Rule-based Enhancement

## Usage

1. Install the requirements.
    ```
    pip install -r requirements.txt
    ```

2. Extract the files to the root directory of the official Snake3v3 code provided by JIDI. The structure of files should be like this.
    ```
    - your working directory
    |- env		            # JIDI official provided
    |- utils	            # JIDI official provided
    |- agents	            # opponents
    |- src
    |- requirements.txt
    |- run.sh
    |- ...                  # other directories and files
    ```

3. Run the following script for training.
    ```
    bash run.sh
    ```

**Note**: some agents as opponents are not included in this repository, please train models for them and remove the corresponding comments in *src/model/opponent_pool.py* to achieve better performance.

## Links
- Credits: The framework of code is based on [Lizhi's code](https://github.com/Lizhi-sjtu/DRL-code-pytorch).

- Competition page: [JIDI Snakes 3v3](http://www.jidiai.cn/env_detail?envid=6).

- JIDI Snakes 3v3 official example: [JIDI AI](https://github.com/jidiai/Competition_3v3snakes).