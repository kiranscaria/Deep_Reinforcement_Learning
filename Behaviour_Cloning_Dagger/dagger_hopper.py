#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20


"""

import pickle
import numpy as np
import tf_util
import tensorflow as tf
import gym
import load_policy

import argparse
import Hopperv1 
import torch
from pathlib import Path
import matplotlib.pyplot as plt

ITERATIONS = 5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    with tf.Session():
        tf_util.initialize()
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break

        expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}

    print('Expert Data')
    print('Observations : ', expert_data['observations'].shape)
    print('Actions : ', expert_data['actions'].shape)

    # Step 1 : Training Policy from human data
    _, _ = Hopperv1.train(expert_data)
 
    for k in range(ITERATIONS):
        # Step 2 : Run the trained policy
        with tf.Session():
            tf_util.initialize()
            observations = []
            actions = []
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                steps = 0
                while not done:
                    action = Hopperv1.test(obs[None,:])
                    observations.append(obs)
                    obs, r, done, _ = env.step(action)
                    steps += 1
                    if args.render:
                        env.render()
                    action = policy_fn(obs[None,:])
                    actions.append(action)
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break

            policy_data = {'observations': np.array(observations),
                           'actions': np.array(actions)}

        print('Policy Data')
        print('Observations : ', policy_data['observations'].shape)
        print('Actions : ', policy_data['actions'].shape)

        # Step 3 : Merging both the data
        augmented_data = {'observations': np.concatenate((expert_data['observations'],policy_data['observations']), axis=0),
                      'actions': np.concatenate((expert_data['actions'], policy_data['actions']), axis=0)}

        # Step 4 : Retrain
        epoch, loss = Hopperv1.train(augmented_data)
        print('Epoch : ', epoch)
        print('Loss : ', loss)

        plt.plot(epoch, loss, label=k)
        plt.title('Hopper-v1')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
