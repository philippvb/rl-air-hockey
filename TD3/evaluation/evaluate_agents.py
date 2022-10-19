import sys
import time

import laserhockey.hockey_env as h_env
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

sys.path.append("P:/Dokumente/3 Uni/WiSe2021/Reinforcement_Learning/Reinforcement_Learning_Winter2021/FinalProject/TD3/src")
from agent import TD3Agent
from utils import HiddenPrints


def evaluate(hyperparameters):
    """Evaluates an agent

    Args:
        hyperparameters (dictionary): The hyperparameters given as a dictionary
    """

    # Init agent and enviroment
    load_path = hyperparameters["load_path"]
    state_dim=18
    action_dim=4
    max_action=1
    iterations=hyperparameters["max_iterations"]
    episodes=hyperparameters["episodes"]
    train_mode = hyperparameters["train_mode"]

    agent1 = TD3Agent([state_dim + action_dim, 256, 256, 1],
                        [state_dim, 256, 256, action_dim])

    agent1.load(load_path)
    if hyperparameters["self_play"]:
        print(f"Evaluating agent {load_path} against himself.")
        agent2=agent1
    else:
        agent2 = h_env.BasicOpponent(weak=hyperparameters["weak"])
        print(f"Evaluating agent {load_path} against a {'weak' if hyperparameters['weak'] else 'hard'} agent")

    if train_mode == "defense":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    elif train_mode == "shooting":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
    else:
        env = h_env.HockeyEnv()

    # performance tracking
    total_rewards = []
    wins_losses=[]
    
    # run enviroment and track performance
    for episode_count in tqdm(range(episodes)):
        env.reset()
        episode_reward=0
        for i in range(iterations):
            if hyperparameters["render"]:
                env.render()
                time.sleep(0.001)

            a1 = agent1.act(env.obs_agent_two())
            a2 = agent2.act(env.obs_agent_two())
            with HiddenPrints():
                obs, r, d, info = env.step(np.hstack([a1,a2]))
            episode_reward+=r
            if d:
                total_rewards.append(episode_reward)
                wins_losses.append(info["winner"])
                env.reset()
                break

    print(f"The agent has a mean reward of {np.mean(total_rewards)}, a win ratio of {wins_losses.count(1)/episodes} and a loss ration of {wins_losses.count(-1)/episodes} and a tie ratio of {wins_losses.count(0)/episodes}")


def plot_performance(network_path):
    """Plots the perforamnce.csv file from the given directory

    Args:
        network_path (String): directory to performance file
    """
    df = pd.read_csv(network_path + "/performance.csv")
    plt.figure()
    plt.subplot(2,1,1)
    x = list(range(len(df.index)))
    coef = np.polyfit(x, df["Episode_critic_loss"],1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(df["Episode_critic_loss"])
    # average
    plt.plot(df["Episode_critic_loss"].rolling(window=100).mean())
    # polynomial fit
    plt.plot(poly1d_fn(list(range(len(df.index)))))


    plt.subplot(2,1,2)
    x = list(range(len(df.index)))
    coef = np.polyfit(x, df["Episode_rewards"],1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(df["Episode_rewards"])
    plt.plot(df["Episode_rewards"].rolling(window=100).mean())
    plt.plot(poly1d_fn(list(range(len(df.index)))))
    plt.show()



#-------------------- Example configuration -----------------------------------
config = {
    "load_path": "FinalProject/TD3/networks/closeness/TD3_baseline",
    "max_iterations": 1000,
    "episodes":1000,
    "render":False,
    "train_mode":"normal",
    "weak":True,
    "self_play":True
}

# evaluate(config)
# plot_performance("FinalProject/TD3/networks/closeness/TD3_baseline")
