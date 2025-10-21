# -*- coding: utf-8 -*-


from stable_baselines3 import PPO, A2C, DQN
import os
from drone_environment import droneEnv
import time
from datetime import datetime


# Load the trained agent
model_path = f"Training/Models/07-22-2024-18-30-30/2000000"
# model=PPO.load(model_path, env=env)
# model=DQN.load(model_path, env=env)
env = droneEnv(observation_mode='disc', action_mode='cont', render=True)
model = A2C.load(model_path, env=env)

episodes = 100
idxs = []

for ep in range(episodes):
    obs=env.reset()
    # idxs.append(env.idx)
    done=False
    # xs = []
    # ys = []
    while not done:
        action, _= model.predict(obs)
        obs, reward, done, _, info = env.step(action)

        print(reward)

models_dir = f"Training/Models/{int(time.time())}/"
logdir = f"Training/Logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# env = droneEnv('cont', render=True)
env = droneEnv(observation_mode='disc', action_mode='cont', render=True)
now = datetime.now()

# It will check your custom environment and output additional warnings if needed
# check_env(env)
date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
print(date_time)
models_dir = f"Training/Models/{date_time}/"
logdir = f"Training/Logs/{date_time}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# env.reset()

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
# model = DQN('MlpPolicy', env, verbose=1, buffer_size=5000, learning_starts=1000 ,tensorboard_log=logdir)

TIMESTEPS = 1000000
iters = 1

while iters<100:
    iters += 1
    print('iteration: ', iters)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
