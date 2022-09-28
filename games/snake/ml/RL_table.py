"""
The template of the script for playing the game in the ml mode
"""
import math
import numpy as np
import sys
import random

'''
絕對方向 : (UP, RIGHT, DOWN, LEFT) = (0, 1, 2, 3)
相對方向 : (LEFT, STRAIGHT, RIGHT) = (0, 1, 2)
'''


def dir_2_num(direction):
    if direction == "UP":
        return 0
    if direction == "DOWN":
        return 2
    if direction == "LEFT":
        return 3
    if direction == "RIGHT":
        return 1


def num_2_dirStr(direction):
    if direction == 0:
        return "UP"
    if direction == 2:
        return "DOWN"
    if direction == 3:
        return "LEFT"
    if direction == 1:
        return "RIGHT"


def get_relative_dir(pre_dir, dir):
    """
    input should be string, like "UP"、"LEFT"...
    return dir's relative direction of pre_dir
    """
    d1 = dir_2_num(pre_dir)
    d2 = dir_2_num(dir)
    d = d1 - d2
    if d == -3 or d == 1:
        return 0
    if d == 0:
        return 1
    if d == -1 or d == 3:
        return 2


def relative_2_absolute(pre_dir, dir):
    """
    pre_dir use absolute dirction in string ("UP"..)
    dir use relative dirction in number (0,1...)
    return absolute direction of dir in number
    """
    pd = dir_2_num(pre_dir)
    return (pd + dir - 1) % 4


def obstacle(point, body):
    """
    Judge that if the point is the block or wall
    """
    if point[0] < 0 or point[0] >= 300 or point[1] < 0 or point[1] >= 300:
        return 1
    else:
        for b in body:
            if point[0] == b[0] and point[1] == b[1]:
                return 1
    return 0


def get_state(pre_direction, scene_info):
    head_x = scene_info["snake_head"][0]
    head_y = scene_info["snake_head"][1]
    food_x = scene_info["food"][0]
    food_y = scene_info["food"][1]
    body = scene_info["snake_body"]
    state = []
    state.append(dir_2_num(pre_direction))
    if pre_direction == "UP":
        if food_y > head_y:
            state.append(0)
        elif food_y < head_y:
            state.append(2)
        else:
            state.append(1)
        if food_x < head_x:
            state.append(0)
        elif food_x == head_x:
            state.append(1)
        else:
            state.append(2)
        state.append(obstacle((head_x-10, head_y), body))
        state.append(obstacle((head_x, head_y-10), body))
        state.append(obstacle((head_x+10, head_y), body))
    elif pre_direction == "RIGHT":
        if food_x < head_x:
            state.append(0)
        elif food_x > head_x:
            state.append(2)
        else:
            state.append(1)
        if food_y < head_y:
            state.append(0)
        elif food_y == head_y:
            state.append(1)
        else:
            state.append(2)
        state.append(obstacle((head_x, head_y-10), body))
        state.append(obstacle((head_x+10, head_y), body))
        state.append(obstacle((head_x, head_y+10), body))
    elif pre_direction == "DOWN":
        if food_y < head_y:
            state.append(0)
        elif food_y > head_y:
            state.append(2)
        else:
            state.append(1)
        if food_x > head_x:
            state.append(0)
        elif food_x == head_x:
            state.append(1)
        else:
            state.append(2)
        state.append(obstacle((head_x+10, head_y), body))
        state.append(obstacle((head_x, head_y+10), body))
        state.append(obstacle((head_x-10, head_y), body))
    elif pre_direction == "LEFT":
        if food_x > head_x:
            state.append(0)
        elif food_x < head_x:
            state.append(2)
        else:
            state.append(1)
        if food_y > head_y:
            state.append(0)
        elif food_y == head_y:
            state.append(1)
        else:
            state.append(2)
        state.append(obstacle((head_x, head_y+10), body))
        state.append(obstacle((head_x-10, head_y), body))
        state.append(obstacle((head_x, head_y-10), body))
    return tuple(state)


def get_state_new(pre_direction, scene_info):
    head_x = scene_info["snake_head"][0]
    head_y = scene_info["snake_head"][1]
    food_x = scene_info["food"][0]
    food_y = scene_info["food"][1]
    body = scene_info["snake_body"]
    state = []
    state.append(dir_2_num(pre_direction))
    if pre_direction == "UP":
        if food_y > head_y:
            state.append(0)
        elif food_y < head_y:
            state.append(2)
        else:
            state.append(1)
        if food_x < head_x:
            state.append(0)
        elif food_x == head_x:
            state.append(1)
        else:
            state.append(2)
        state.append(obstacle((head_x-10, head_y+10), body))
        state.append(obstacle((head_x-10, head_y), body))
        state.append(obstacle((head_x-10, head_y-10), body))
        state.append(obstacle((head_x, head_y-10), body))
        state.append(obstacle((head_x+10, head_y-10), body))
        state.append(obstacle((head_x+10, head_y), body))
        state.append(obstacle((head_x+10, head_y+10), body))
    elif pre_direction == "RIGHT":
        if food_x < head_x:
            state.append(0)
        elif food_x > head_x:
            state.append(2)
        else:
            state.append(1)
        if food_y < head_y:
            state.append(0)
        elif food_y == head_y:
            state.append(1)
        else:
            state.append(2)
        state.append(obstacle((head_x-10, head_y-10), body))
        state.append(obstacle((head_x, head_y-10), body))
        state.append(obstacle((head_x+10, head_y-10), body))
        state.append(obstacle((head_x+10, head_y), body))
        state.append(obstacle((head_x+10, head_y+10), body))
        state.append(obstacle((head_x, head_y+10), body))
        state.append(obstacle((head_x-10, head_y+10), body))
    elif pre_direction == "DOWN":
        if food_y < head_y:
            state.append(0)
        elif food_y > head_y:
            state.append(2)
        else:
            state.append(1)
        if food_x > head_x:
            state.append(0)
        elif food_x == head_x:
            state.append(1)
        else:
            state.append(2)
        state.append(obstacle((head_x+10, head_y-10), body))
        state.append(obstacle((head_x+10, head_y), body))
        state.append(obstacle((head_x+10, head_y+10), body))
        state.append(obstacle((head_x, head_y+10), body))
        state.append(obstacle((head_x-10, head_y+10), body))
        state.append(obstacle((head_x-10, head_y), body))
        state.append(obstacle((head_x-10, head_y-10), body))
    elif pre_direction == "LEFT":
        if food_x > head_x:
            state.append(0)
        elif food_x < head_x:
            state.append(2)
        else:
            state.append(1)
        if food_y > head_y:
            state.append(0)
        elif food_y == head_y:
            state.append(1)
        else:
            state.append(2)
        state.append(obstacle((head_x+10, head_y+10), body))
        state.append(obstacle((head_x, head_y+10), body))
        state.append(obstacle((head_x-10, head_y+10), body))
        state.append(obstacle((head_x-10, head_y), body))
        state.append(obstacle((head_x-10, head_y-10), body))
        state.append(obstacle((head_x, head_y-10), body))
        state.append(obstacle((head_x+10, head_y-10), body))
    return tuple(state)


def get_action(state, q_table, epsilon):
    if np.random.random_sample() < epsilon:  # 有 ε 的機率會選擇隨機 action
        return -1
    else:  # 其他時間根據現有 policy 選擇 action，也就是在 Q table 裡目前 state 中，選擇擁有最大 Q value 的 action
        return np.argmax(q_table[state])


def get_reward(scene_info, pre_len):
    if scene_info["status"] == "GAME_OVER":
        return -700
    elif pre_len < len(scene_info["snake_body"]):
        return 200
    else:
        return 0


def get_epsilon(i, m_i):
    # epsilon-greedy; 隨時間遞減
    i = int(i / (m_i/200))
    return max(0.01, min(1, 1.0 - math.log10((i+1)/25)))


def get_lr(i, m_i):
    # learning rate; 隨時間遞減
    i = int(i / (m_i/200))
    return max(0.01, min(0.5, 1.0 - math.log10((i+1)/25)))


def rule_base(pre_dir, scene_info):
    head_x = scene_info["snake_head"][0]
    head_y = scene_info["snake_head"][1]
    food_x = scene_info["food"][0]
    food_y = scene_info["food"][1]
    if pre_dir == "UP":
        if food_y < head_y:
            return 1
        else:
            if food_x < head_x:
                return 0
            elif food_x > head_x:
                return 2
            else:
                if random.randint(0, 1) == 0:
                    return 0
                else:
                    return 2
    elif pre_dir == "RIGHT":
        if food_x > head_x:
            return 1
        else:
            if food_y < head_y:
                return 0
            elif food_y > head_y:
                return 2
            else:
                if random.randint(0, 1) == 0:
                    return 0
                else:
                    return 2
    elif pre_dir == "DOWN":
        if food_y > head_y:
            return 1
        else:
            if food_x > head_x:
                return 0
            elif food_x < head_x:
                return 2
            else:
                if random.randint(0, 1) == 0:
                    return 0
                else:
                    return 2
    elif pre_dir == "LEFT":
        if food_x < head_x:
            return 1
        else:
            if food_y > head_y:
                return 0
            elif food_y < head_y:
                return 2
            else:
                if random.randint(0, 1) == 0:
                    return 0
                else:
                    return 2


def old_way(state, old_table):
    return np.argmax(old_table[state])


n_buckets = (4, 3, 3, 2, 2, 2, 2, 2, 2, 2)
n_actions = 3

old_table = np.load("q_table_1.npy")
q_table = np.load("q_table_1.npy")
# q_table = np.zeros(n_buckets + (n_actions,))


class MLPlay:
    def __init__(self):
        """
        Constructor
        """
        self.max_episode = 300

        self.i_episode = 0
        self.lr = get_lr(self.i_episode, self.max_episode)
        self.epsilon = get_epsilon(self.i_episode, self.max_episode)
        self.gm = 0.99  # gamma

        self.pre_dir = "DOWN"
        self.pre_dir_re = 1
        self.pre_sta = None
        self.pre_len = 3

        self.rewards = 0

    def update(self, scene_info):
        """
        Generate the command according to the received scene information
        """
        state = get_state_new(self.pre_dir, scene_info)
        if self.pre_sta != None:
            q_max = np.amax(q_table[state])
            reward = get_reward(scene_info, self.pre_len)
            # print(self.pre_sta)
            self.rewards += reward
            q_table[self.pre_sta + (self.pre_dir_re,)] += self.lr * (
                reward + self.gm * q_max - q_table[self.pre_sta + (self.pre_dir_re,)])

        if scene_info["status"] == "GAME_OVER":
            print("In {} episode, totaly used {} frames. and get {} point".format(
                self.i_episode, scene_info["frame"], self.rewards))
            return "RESET"
        # print(state)
        action = get_action(state, q_table, self.epsilon)
        if action == -1:
            action = old_way(state, old_table)

        self.pre_dir = num_2_dirStr(relative_2_absolute(self.pre_dir, action))
        self.pre_dir_re = action
        self.pre_sta = state
        self.pre_len = len(scene_info["snake_body"])
        return self.pre_dir

    def reset(self):
        """
        Reset the status if needed
        """
        if self.i_episode == self.max_episode:
            np.save("q_table_2", q_table)
        self.pre_dir = "DOWN"
        self.i_episode += 1
        self.lr = get_lr(self.i_episode, self.max_episode)
        self.epsilon = get_epsilon(self.i_episode, self.max_episode)
        self.pre_sta = None
        self.rewards = 0
