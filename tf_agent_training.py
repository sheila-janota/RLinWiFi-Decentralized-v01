
#%%
from ns3gym import ns3env
from comet_ml import Experiment, Optimizer
import tqdm
import subprocess
from collections import deque
import numpy as np

from agents.dqn.agent import Agent, Config
from agents.dqn.model import QNetworkTf
from agents.teacher import Teacher, EnvWrapper
from preprocessor import Preprocessor

#%%
#scenario = "convergence"
scenario = "basic"

simTime = 60 # seconds
stepTime = 0.01  # seconds
history_length = 300

EPISODE_COUNT = 15
steps_per_ep = int(simTime/stepTime)

sim_args = {
    "simTime": simTime,
    "envStepTime": stepTime,
    "historyLength": history_length,
    "agentType": Agent.TYPE,
    "scenario": "basic", #"convergence",
    "nWifi": 2, #15,
}

print("Steps per episode:", steps_per_ep)

threads_no = 1
env = EnvWrapper(threads_no, **sim_args)

#%%
env.reset()
ob_space = env.observation_space
ac_space = env.action_space

print("Observation space shape:", ob_space)
print("Action space shape:", ac_space)

assert ob_space is not None

#%%
teacher_0 = Teacher(env, 1, Preprocessor(False))
#teacher_1 = Teacher(env, 1, Preprocessor(False))

lr = 4e-4
config = Config(buffer_size=3*steps_per_ep*threads_no, batch_size=32, gamma=0.7, tau=1e-3, lr=lr, update_every=1)
agent_0 = Agent(QNetworkTf, history_length, action_size=7, config=config)
#agent_1 = Agent(QNetworkTf, history_length, action_size=7, config=config)

agent_0.set_epsilon(0.9, 0.001, EPISODE_COUNT-2)
#agent_1.set_epsilon(0.9, 0.001, EPISODE_COUNT-2)

# Test the model
hyperparams = {**config.__dict__, **sim_args}
tags = ["Rew: normalized speed",
        "Final",
        f"{Agent.NAME}",
        sim_args['scenario'],
        f"LR: {lr}",
        f"Instances: {threads_no}",
        f"Station count: {sim_args['nWifi']}",
        *[f"{key}: {sim_args[key]}" for key in list(sim_args)[:3]]]
# # agent.save()
logger = teacher_0.train(agent_0, EPISODE_COUNT,
                        simTime=simTime,
                        stepTime=stepTime,
                        history_length=history_length,
                        send_logs=True,
                        experimental=True,
                        tags=tags,
                        parameters=hyperparams)
#logger = teacher.eval(agent,
#                        simTime=simTime,
#                        stepTime=stepTime,
#                        history_length=history_length,
#                        tags=tags,
#                        parameters=hyperparams)
# agent.save()

#logger = teacher_1.train(agent_1, EPISODE_COUNT,
#                       simTime=simTime,
#                       stepTime=stepTime,
#                        history_length=history_length,
#                        send_logs=True,
#                        experimental=True,
 #                       tags=tags,
#                        parameters=hyperparams)


# %%
