""" 
 /*
 * @Author: 汤达荣 
 * @Date: 2019-01-10 09:39:15 
 * @Last Modified by: 汤达荣
 * @Last Modified time: 2019-01-10 11:07:15
 * @Email：tdr1991@outlook.com 
 */
""" 
#coding:utf-8 

import time

import numpy as np
import pandas as pd

N_STATES = 6
ACTIONS = ["left", "right"]
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPSODES = 13
FRESH_TIME = 0.3

def build_q_table(states, actions):
    table = pd.DataFrame(np.zeros((states, len(actions))), columns=actions)
    return table

# 这里EPSILON=0.9，就是说90%的时间选择最优策略，10%的时间进行探索
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name

# 这里的逻辑是选出最佳动作的概率设置为EPSILON=0.9，而其它动作一样为1-EPSILON
def make_epsilon_greedy_policy(Q, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * (1 - EPSILON) / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += EPSILON
        return A
    return policy_fn

def policy_decision(state, q_table):
    policy = make_epsilon_greedy_policy(q_table, len(ACTIONS))
    actions_probs = policy(state)
    action_name = np.random.choice(ACTIONS, p=actions_probs)
    return action_name

def get_env_feedback(state, action):
    if action.lower() == "right":
        if state == N_STATES - 2:
             new_state = "terminal"
             reward = 1
        else:
            new_state = state + 1
            reward = 0
    else:
        reward = 0
        if state == 0:
            new_state = state
        else:
            new_state = state - 1
    return new_state, reward

def update_env(state, episode, step_counter):
    env_list = ["-"] * (N_STATES - 1) + ["T"]
    if state == "terminal":
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def q_learing():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPSODES):
        step_counter = 0
        state = 0
        is_terminal = False
        update_env(state, episode, step_counter)
        while not is_terminal:
            #action = policy_decision(state, q_table)
            action = choose_action(state, q_table)
            new_state, reward = get_env_feedback(state, action)
            q_predict = q_table.loc[state, action]
            if new_state != "terminal":
                q_target = reward + GAMMA * q_table.iloc[new_state, :].max()
            else:
                q_target = reward
                is_terminal = True
            q_table.loc[state, action] += ALPHA * (q_target - q_predict)
            update_env(state, episode, step_counter)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = q_learing()
    print('\r\nQ-table:\n')
    print(q_table)