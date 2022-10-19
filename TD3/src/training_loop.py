# The main class for training:
import json
import os
import sys
import time

import laserhockey.hockey_env as h_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from agent import TD3Agent
from utils import HiddenPrints, ReplayBuffer, moving_average


def training_loop(hyperparameters):
    print(f"Starting training with hyperparameters: {hyperparameters}")
    save_path = hyperparameters["save_path"]
    load_path = hyperparameters["load_path"]

    # create the save path and save hyperparameter configuration
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        a = input("Warning, Directory already exists. Dou want to continue?")
        if a not in ["Y","y"]:
            raise Exception("Path already exists, please start with another path.")

    with open(save_path+ "/parameters.json", "w") as f:
        json.dump(hyperparameters, f)

    # general configurations
    state_dim=18
    action_dim=4
    max_action=1
    iterations=hyperparameters["max_iterations"]
    batch_size=hyperparameters["batch_size"]
    max_episodes=hyperparameters["max_episodes"]
    train_mode = hyperparameters["train_mode"]
    closeness_factor=hyperparameters["closeness_factor"]
    c = closeness_factor

    # init the agent
    agent1 = TD3Agent([state_dim + action_dim, 256, 256, 1],
                        [state_dim, 256, 256, action_dim],
                        optimizer=hyperparameters["optimizer"],
                        policy_noise=hyperparameters["policy_noise"],
                        policy_noise_clip=hyperparameters["policy_noise_clip"],
                        gamma=hyperparameters["gamma"],
                        delay=hyperparameters["delay"],
                        tau=hyperparameters["tau"],
                        lr=hyperparameters["lr"],
                        max_action=max_action,
                        weight_decay=hyperparameters["weight_decay"])

    # load the agent if given
    loaded_state=False
    if load_path:
        agent1.load(load_path)
        loaded_state=True

    # define opponent
    if hyperparameters["self_play"]:
        agent2=agent1
    else:
        agent2 = h_env.BasicOpponent(weak=hyperparameters["weak_agent"])

    # load enviroment and replaybuffer
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    if train_mode == "defense":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    elif train_mode == "shooting":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
    else:
        env = h_env.HockeyEnv()


    # add figure to plot later
    if hyperparameters["plot_performance"]:
        fig, (ax_loss, ax_reward) = plt.subplots(2)
        ax_loss.set_xlim(0, max_episodes)
        ax_loss.set_ylim(0, 20)
        ax_reward.set_xlim(0, max_episodes)
        ax_reward.set_ylim(-30, 20)

    with HiddenPrints():
    # first sample enough data to start:
        obs_last = env.reset()
        for i in range(batch_size*100):
            a1 = env.action_space.sample()[:4] if not loaded_state else agent1.act(env.obs_agent_two())
            a2 = agent2.act(env.obs_agent_two())
            obs, r, d, info = env.step(np.hstack([a1,a2]))
            done = 1 if d else 0
            replay_buffer.add(obs_last, a1, obs, r, done)
            obs_last=obs
            if d:
                obs_last = env.reset()

    print("Finished collection of data prior to training")

    # tracking of performance
    episode_critic_loss=[]
    episode_rewards=[]
    win_count=[]
    if not os.path.isfile(save_path + "/performance.csv"):
        pd.DataFrame(data={"Episode_rewards":[], "Episode_critic_loss":[], "Win/Loss":[]}).to_csv(save_path + "/performance.csv", sep=",", index=False)

    # Then start training
    for episode_count in range(max_episodes+1):
        obs_last = env.reset()
        total_reward=0
        critic_loss=[]

        for i in range(iterations):
            # run the enviroment
            with HiddenPrints():
                with torch.no_grad():
                    a1 =  agent1.act(env.obs_agent_two()) + np.random.normal(loc=0, scale=hyperparameters["exploration_noise"], size=action_dim)
                a2 = agent2.act(env.obs_agent_two())
                obs, r, d, info = env.step(np.hstack([a1,a2]))
            total_reward+=r
            done = 1 if d else 0

            # mopify reward with cloeness to puck reward
            if hyperparameters["closeness_decay"]:
                c = closeness_factor *(1 - episode_count/max_episodes)
            newreward = r + c * info["reward_closeness_to_puck"] 

            # add to replaybuffer
            replay_buffer.add(obs_last, a1, obs, newreward, done)
            obs_last=obs
            
            # sample minibatch and train
            states, actions, next_states, reward, done = replay_buffer.sample(batch_size)
            loss = agent1.train(states, actions, next_states, reward, done)
            critic_loss.append(loss.detach().numpy())

            # if done, finish episode
            if d:
                episode_rewards.append(total_reward)
                episode_critic_loss.append(np.mean(critic_loss))
                win_count.append(info["winner"])
                print(f"Episode {episode_count} finished after {i} steps with a total reward of {total_reward}")
                
                # Online plotting
                if hyperparameters["plot_performance"] and episode_count>40 :
                    ax_loss.plot(list(range(-1, episode_count-29)), moving_average(episode_critic_loss, 30), 'r-')
                    ax_reward.plot(list(range(-1, episode_count-29)), moving_average(episode_rewards, 30), "r-")
                    plt.draw()
                    plt.pause(1e-17)

                break
        
        # Intermediate evaluation of win/loss and saving of model
        if episode_count % 500 ==0 and episode_count != 0:
                print(f"The agents win ratio in the last 500 episodes was {win_count[-500:].count(1)/500}")
                print(f"The agents loose ratio in the last 500 episodes was {win_count[-500:].count(-1)/500}")
                try:
                    agent1.save(save_path)
                    print("saved model")
                except Exception:
                    print("Saving Failed model failed")
                pd.DataFrame(data={"Episode_rewards": episode_rewards[-500:], "Episode_critic_loss": episode_critic_loss[-500:], "Win/Loss": win_count[-500:]}).to_csv(save_path + "/performance.csv", sep=",", index=False, mode="a", header=False)
                    
    print(f"Finished training with a final mean reward of {np.mean(episode_rewards[-500:])}")





    # plot the performance summary
    if hyperparameters["plot_performance_summary"]:
            try:
                fig, (ax1, ax2) = plt.subplots(2)
                x = list(range(len(episode_critic_loss)))
                coef = np.polyfit(x, episode_critic_loss,1)
                poly1d_fn = np.poly1d(coef)
                ax1.plot(episode_critic_loss)
                ax1.plot(poly1d_fn(list(range(len(episode_critic_loss)))))


                x = list(range(len(episode_rewards)))
                coef = np.polyfit(x, episode_rewards,1)
                poly1d_fn = np.poly1d(coef)
                ax2.plot(episode_rewards)
                ax2.plot(poly1d_fn(list(range(len(episode_rewards)))))
                fig.show()
                fig.savefig(save_path + "/performance.png", bbox_inches="tight")
            except:
                print("Failed saving figure")
        

    

#-------------------- Example configurations ---------------------------------
# The configurations can be run by calling training_loop(your_configuration)

# The TD3 baseline with the configuration from the authors
TD3_baseline = {
    "save_path": "FinalProject/TD3/networks/TD3_baseline",
    "load_path": None, 
    "max_iterations": 1000,
    "batch_size": 256, 
    "max_episodes":10000,
    "plot_performance": False,
    "train_mode": "normal",
    "plot_performance_summary": True,
    "closeness_factor":0,
    "closeness_decay":False, 
    "lr": 3e-4, 
    "optimizer": "Adam",
    "weight_decay": 1e-2,
    "policy_noise": 0.1, 
    "policy_noise_clip": 0.5,
    "gamma": 0.9,
    "delay": 2, 
    "tau": 0.005,
    "exploration_noise":0.1, 
    "weak_agent": True,
    "self_play":False,
}

# TD3 with Reward modification
TD3_reward_modification = {
    "save_path": "FinalProject/TD3/networks/TD3_reward_modification",
    "load_path": None, 
    "max_iterations": 1000,
    "batch_size": 256, 
    "max_episodes":10000,
    "plot_performance": False,
    "train_mode": "normal",
    "plot_performance_summary": True,
    "closeness_factor":10, # modify the reward
    "closeness_decay":False, 
    "lr": 3e-4, 
    "optimizer": "Adam",
    "weight_decay": 1e-2,
    "policy_noise": 0.1, 
    "policy_noise_clip": 0.5,
    "gamma": 0.9,
    "delay": 2, 
    "tau": 0.005,
    "exploration_noise":0.1, 
    "weak_agent": True,
    "self_play":False,
}

# Self-Play agent, pretrained with TD3_Reward_modification
TD3_self_play = {
    "save_path": "FinalProject/TD3/networks/TD3_self_play",
    "load_path": "FinalProject/TD3/networks/TD3_reward_modification", 
    "max_iterations": 1000,
    "batch_size": 256, 
    "max_episodes":10000,
    "plot_performance": False,
    "train_mode": "normal",
    "plot_performance_summary": True,
    "closeness_factor":0,
    "closeness_decay":False, 
    "lr": 3e-4, 
    "optimizer": "Adam",
    "weight_decay": 1e-2,
    "policy_noise": 0.1, 
    "policy_noise_clip": 0.5,
    "gamma": 0.9,
    "delay": 2, 
    "tau": 0.005,
    "exploration_noise":0.1, 
    "weak_agent": True,
    "self_play":True, # use self play
}

#training_loop(TD3_baseline)

