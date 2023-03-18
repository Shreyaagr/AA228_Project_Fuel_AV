"""
This Code is partially adapted from: https://github.com/yanxi0830/CS238CrosswalkDriving/blob/main/src/jaywalker/QLearning.py
"""

import sys
import numpy as np
import pandas as pd
import random
import math
import mpmath
import time
from config import CONFIG
from simenv import *
import matplotlib.pyplot as plt
from collections import *

random.seed(110)
np.random.seed(110)

def write_policy(pol, filename):
    with open(filename+".policy", 'w') as f:
        for p in pol:
            f.write("{}\n".format(p))
    with open(filename+".txt", 'w') as f:
        for p in pol:
            f.write("{}\n".format(p))

            
def extract_policy(Qmat):
    print('policy file n_rows non-random: ', np.count_nonzero(np.sum(Qmat, axis=1)))
    pol = 1 + np.argmax(Qmat, axis=1)
    return pol


def plot(x, list_y, x_lab, y_lab, title, label_var, save_path, ylim=False):
    # Plot dataset
    fig = plt.figure()
    for j in range(len(list_y)):
        plt.plot(x, list_y[j], linewidth=2, label=label_var[j])
    # Add labels and save to disk
    plt.legend()
    #plt.grid('on')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    if ylim:
        plt.ylim([-1,7.0])
    plt.title(title)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300) 
    

class Qlearning:
    def __init__(self, env): # find alternative to env definition
        self.env = env 
        self.statespace = env.observation_space
        self.actionspace = env.action_space.n # [acceleration (del_v): -1,0,1]
        self.lr = 0.7
        self.discountfactor = 0.9
        self.episode = 12000
        self.horizon = 1000
        self.eps = 0.8
        self.decay_factor = 0.9
        position = int(env.max_position + 1)
        v = int(env.max_velocity - env.min_velocity + 1)
        rg = int(env.max_road_grade - env.min_road_grade + 1) # possible number of road grade values
        self.Qmat = np.zeros((position * v * rg, self.actionspace))

    def state_to_index(self, state):
        # i, j, k == > i * (maxj + 1) * (maxk + 1) + j * (maxk + 1) + k
        max_i = self.env.max_position
        max_j = self.env.max_velocity
        max_k = self.env.max_road_grade
        i = state[0]
        j = state[1]
        k = state[2]
        index = i * (max_j + 1) * (max_k + 1) + j * (max_k + 1) + k
        return index

    def act(self, state, deterministic):
        if deterministic:
            action = np.argmax(self.Qmat[state, :])
        else:
            epsgreedy = random.uniform(0, 1)
            if epsgreedy > self.eps:
                action = np.argmax(self.Qmat[state, :])
            else:
                self.eps *= self.decay_factor
                action = np.random.randint(self.actionspace, size=1, dtype=int)[0]

        return action

    def update(self, curstate, curaction, reward, nextstate):
        nextu = np.amax(self.Qmat[nextstate])
        self.Qmat[curstate][curaction] += self.lr * (
                reward + self.discountfactor * nextu - self.Qmat[curstate][curaction])

    def train(self, rr=0):  
        rewardperep = []
        for i in range(self.episode):
            state = self.state_to_index(self.env.reset())
            totalreward = 0
            for j in range(self.horizon):
                action = self.act(state, deterministic=False)
                next_state, reward, done, info = self.env.step(action)
                next_state = self.state_to_index(next_state)
                self.update(state, action, reward, next_state)
                totalreward += reward
                state = next_state
                if done:
                    break
            rewardperep.append(totalreward)
        return rewardperep, np.mean(np.array(rewardperep[-50:])) # mean total reward for last 50 episodes


def test(env, model, n_steps, random_policy=False):
    # Test
    env.reset()
    state = model.state_to_index(env.state)

    total_reward = 0
    step_to_goal = 0
    success = False
    state_list = [[env.state[0], env.state[1], env.state[2]]]
    
    if random_policy==True:
        for step in range(n_steps):
            action = np.random.randint(env.action_space.n, size=1, dtype=int)[0]
            # print("Step {}".format(step + 1))
            # print("Action: ", env.actions_list[action])
            state, reward, done, info = env.step(action, eval_mode=True)
            state_list.append([state[0], state[1], state[2]])
            state = model.state_to_index(state)
            total_reward += reward
    
            # print('obs=', state, 'current reward=', reward, 'total reward=', total_reward, 'done=', done)
            # env.render(mode='console')
            if done:
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                # print("Goal reached!", "reward=", reward, 'total reward=', total_reward)
                step_to_goal = step +1
                if reward == CONFIG["fixed_rewards_dict"]["reached_goal"]:
                    success = True
                break
    else:
        for step in range(n_steps):
            action = model.act(state, deterministic=True)
            # print("Step {}".format(step + 1))
            # print("Action: ", env.actions_list[action])
            state, reward, done, info = env.step(action, eval_mode=True)
            state_list.append([state[0], state[1], state[2]])
            state = model.state_to_index(state)
            total_reward += reward
    
            # print('obs=', state, 'current reward=', reward, 'total reward=', total_reward, 'done=', done)
            # env.render(mode='console')
            if done:
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                # print("Goal reached!", "reward=", reward, 'total reward=', total_reward)
                step_to_goal = step +1
                if reward == CONFIG["fixed_rewards_dict"]["reached_goal"]:
                    success = True
                break
    return (total_reward, step_to_goal, int(success == True), state_list)


def main():
    random_runs_q = defaultdict(list) # qlearning
    random_runs_r = defaultdict(list) # random actions

    rewardperep_random_train = []
    n_steps = 10000
    RANDOM_RUNS = 5
    best_train_reward = -10000.
    
    # Train
    # finding best policy using RANDOM_RUNS re-trainings using random seed
    # first training
    env = SimEnv(CONFIG)
    best_model = Qlearning(env)
    rewardperep, best_train_reward = best_model.train(rr=0)
    rewardperep_random_train.append(np.array(rewardperep))
    best_rr           = 0
    
    # remaining RANDOM_RUNS-1 re-trainings
    for rr in range(1, RANDOM_RUNS):
        # Train
        env = SimEnv(CONFIG)
        model = Qlearning(env)
        rewardperep, train_reward = model.train(rr)
        rewardperep_random_train.append(np.array(rewardperep))
        if train_reward > best_train_reward:
            best_train_reward = train_reward
            best_model        = model
            best_rr           = rr
    print('Best training random-run: ', best_rr+1)
    plot(np.array([i for i in range(rewardperep_random_train[0].shape[0])]), rewardperep_random_train, 'Episode number', 
         'Total reward', 'Training reward vs episode', ['run {}'.format(i+1) for i in range(len(rewardperep_random_train))], 
         'results/qlearning_rewardperep_training.png')
 
    
    # Test
    # evaluating best model using RANDOM_RUNS re-tests of qlearning
    model = best_model
    for rr in range(RANDOM_RUNS):
        total_reward, step_to_goal, succ, state_list_q = test(env, model, n_steps)
        random_runs_q['total_reward'].append(total_reward)
        random_runs_q['steps'].append(step_to_goal)
        random_runs_q['success'].append(succ)
    print('random_runs_q: ', random_runs_q)
    # calculate average total reward
    avg_reward = sum(random_runs_q['total_reward']) / RANDOM_RUNS
    print("Average Total Reward (q learning): {}".format(avg_reward))
    avg_steps = sum(random_runs_q['steps']) / RANDOM_RUNS
    print("Average Steps (q learning): {}".format(avg_steps))
    sucess_rate = sum(random_runs_q['success']) / RANDOM_RUNS
    print("Success Rate (q learning): {}".format(sucess_rate))
    print("----------")
 
    env.reset()
    # Test on random policy
    # random policy evaluation
    for rr in range(RANDOM_RUNS):
        total_reward, step_to_goal, succ, state_list_r = test(env, model, n_steps, random_policy=True)
        random_runs_r['total_reward'].append(total_reward)
        random_runs_r['steps'].append(step_to_goal)
        random_runs_r['success'].append(succ)
    print('random_runs_r: ', random_runs_r)
    # calculate average total reward
    avg_reward = sum(random_runs_r['total_reward']) / RANDOM_RUNS
    print("Average Total Reward (random policy): {}".format(avg_reward))
    avg_steps = sum(random_runs_r['steps']) / RANDOM_RUNS
    print("Average Steps (random policy): {}".format(avg_steps))
    sucess_rate = sum(random_runs_r['success']) / RANDOM_RUNS
    print("Success Rate (random policy): {}".format(sucess_rate))
    
    
    # compare velocity profiles of random policy vs q-learning
    plot([i for i in range(len(state_list_q))], [np.array(state_list_q)[:,1], np.array(state_list_q)[:,-1]],
          'time', 'velocity / road grade', 'velocity and road grade profile vs time',
          ['velocity', 'road grade'], 'results/qlearning_profile', ylim=True)
    plot([i for i in range(len(state_list_r))], [np.array(state_list_r)[:,1], np.array(state_list_r)[:,-1]],
          'time', 'velocity / road grade', 'velocity and road grade profile vs time',
          ['velocity', 'road grade'], 'results/randomlearning_profile', ylim=True)
    print(np.array(state_list_q)[:,-1], np.array(state_list_r)[:,-1])
    
    
if __name__ == '__main__':
    main()

