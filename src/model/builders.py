import math

import torch
import numpy as np

from utils.constants import HEIGHT, WIDTH, ACTION
from src.utils.calc_l1_dist import _c, l1_dist


def state_builder(observations_dicts):
    features = []
    base_board = np.zeros((HEIGHT, WIDTH))
    for snake_id in range(2, 7+1):
        for i, pos in enumerate(observations_dicts[0][snake_id]):
            if i == 0:  # head
                base_board[pos[0], pos[1]] = 3*(snake_id-2)+3
            elif i == len(observations_dicts[0][snake_id])-1:  # tail
                base_board[pos[0], pos[1]] = 3*(snake_id-2)+1
            else:  # body
                base_board[pos[0], pos[1]] = 3*(snake_id-2)+2
    for ob_dict in observations_dicts:
        feature = []
        self_snake_id = ob_dict["controlled_snake_index"]
        teammate, enemy = (2, 3, 4), (5, 6, 7)
        if self_snake_id >= 5:
            teammate, enemy = enemy, teammate
        self_head_pos = ob_dict[self_snake_id][0]
        bean_distribution = distribute_beans(
            [ob_dict[snake][0] for snake in teammate],
            ob_dict[1]
        )
        my_bean_idx = bean_distribution[(self_snake_id-2)%3]
        bean_pos = ob_dict[1][my_bean_idx]
        feature.append(_c(bean_pos[0] - self_head_pos[0], "height"))
        feature.append(_c(bean_pos[1] - self_head_pos[1], "width"))

        self_tail_pos = ob_dict[self_snake_id][-1]
        feature.append(_c(self_tail_pos[0]-self_head_pos[0], "height"))
        feature.append(_c(self_tail_pos[1]-self_head_pos[1], "width"))


        board = base_board.copy()
        board[self_tail_pos[0], self_tail_pos[1]] = 0
        for delta in ((-1,0), (1,0), (0,-1), (0,1)):
            _x = (self_head_pos[0] + delta[0]) % HEIGHT
            _y = (self_head_pos[1] + delta[1]) % WIDTH
            feature.append(3. if (board[_x, _y]>0 and board[_x, _y]%3==0) else board[_x, _y]%3)

        for delta in ((-1,-1), (-1,1), (1,-1), (1,1), (-2,0), (2,0), (0,-2), (0,2)):
            _x = (self_head_pos[0] + delta[0]) % HEIGHT
            _y = (self_head_pos[1] + delta[1]) % WIDTH
            feature.append(3. if (board[_x, _y]>0 and board[_x, _y]%3==0) else board[_x, _y]%3)
        feature.append(len(ob_dict[self_snake_id]))
        features.append(feature)
    return torch.tensor(features, dtype=torch.float)


def distribute_beans(head_pos_list, bean_pos_list):
    dist_mtx = np.zeros((len(head_pos_list), len(bean_pos_list)))
    for i, head_pos in enumerate(head_pos_list):
        for j, bean_pos in enumerate(bean_pos_list):
            delta_x = _c(bean_pos[0] - head_pos[0], "height")
            delta_y = _c(bean_pos[1] - head_pos[1], "width")
            bean_dist = abs(delta_x) + abs(delta_y)
            dist_mtx[i, j] = bean_dist
    distribution = [None for _ in range(len(head_pos_list))]
    for _ in range(len(head_pos_list)):
        index = np.argmin(dist_mtx)
        snake = int(index // len(bean_pos_list))
        bean = index % len(bean_pos_list)
        distribution[snake] = bean
        dist_mtx[snake, :] = 100
        dist_mtx[:, bean] = 100
    return distribution


def mask_builder(obs):
    masked_actions = []
    for ob in obs:
        masked_actions.append(get_mask(ob))
    return masked_actions


def get_mask(ob):
    index = ob["controlled_snake_index"]
    mask = np.zeros(4, dtype=int)
    if len(ob[index]) == 1:
        return mask
    head_x, head_y = ob[index][0]
    neck_x, neck_y = ob[index][1]
    for idx, action in ACTION.items():
        next_head_x = (head_x + action[0]) % HEIGHT
        next_head_y = (head_y + action[1]) % WIDTH
        if next_head_x == neck_x and next_head_y == neck_y:
            mask[idx] = 1
    return mask


def reward_builder(obs, rs):
    rewards = []
    for i, (ob, r) in enumerate(zip(obs, rs)):
        if r > 0:
            head, tail = ob[i+2][0], ob[i+2][-1]
            head_tail_dist = l1_dist(head, tail)
            self_len = len(ob[i+2])
            rewards.append(r - math.pow((self_len-0)*(head_tail_dist-0)/300, 2))
        else:
            rewards.append(r)
    return rewards