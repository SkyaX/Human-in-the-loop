"""
Implementation of TAMER (Knox + Stone, 2009)
When training, use 'W' and 'A' keys for positive and negative rewards
"""

import asyncio
import gym

from tamer.agent import Tamer


async def main():

	env_name = "CartPole-v1" # 'MountainCar-v0'
	env = gym.make(env_name)

	# hyperparameters
	discount_factor = 1
	epsilon = 0  # vanilla Q learning actually works well with no random exploration
	min_eps = 0
	num_episodes = 10
	tame = True  # set to false for vanilla Q learning

	# set a timestep for training TAMER
	# the more time per step, the easier for the human
	# but the longer it takes to train (in real time)
	# 0.2 seconds is fast but doable
	tamer_training_timestep = 5

	agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame,
				tamer_training_timestep, model_file_to_load=None, env_name=env_name)

	reward_training,ep_train= await agent.train(model_file_to_save='autosave')
	agent.play(n_episodes=1, render=True)
	reward_evaluating,ep_eval=agent.evaluate(n_episodes=30)

	for i in range(len(ep_train)) : 
		f.write(str(reward_training[i]) + " "  + (str(ep_train[i])) + "\n")
	f.close()
	for i in range(len(ep_eval)) :
		b.write(str(reward_evaluating[0][i]) + " "  + (str(ep_eval[i])) + "\n")
	b.close()

if __name__ == '__main__':
	f=open('train.txt','w')
	b=open('ev.txt','w')
	asyncio.run(main())
	





