import random

def my_controller(*args, **kwargs):
    action = [0 for _ in range(4)]
    action[random.randint(0, 3)] = 1
    return [action]