"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py. I wrote this file to demonstrate
	that our trained policy exists independently of our learning algorithm,
	which resides in ppo.py. Thus, we can test our trained policy without 
	relying on ppo.py.
"""
import numpy as np
import time
import torch
from torch.distributions import MultivariateNormal
def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(cov_var,cov_mat,policy, env, render):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 

		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.

		Note:
			If you're unfamiliar with Python generators, check this out:
				https://wiki.python.org/moin/Generators
			If you're unfamiliar with Python "yield", check this out:
				https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	"""
	
	# Rollout until user kills process
	while True:
		# obs = env.reset()
		radar_obs,state_obs = env.reset()
		time.sleep(1)
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return
		try :
			while not done and t<=50:
				t += 1

				# Render environment if specified, off by default
				# if render:
				# 	env.render()

				# Query deterministic action from policy and run it
				# action = policy(obs).detach().numpy()
				# action, log_prob = get_action(radar_obs,state_obs,cov_var,cov_mat)
				mean = policy(radar_obs,state_obs,None)
				dist = MultivariateNormal(mean, cov_mat)
				action = dist.sample()
				action.detach().numpy()
				choice = action[0]
				final_action = choose_action_straight(choice)
				time.sleep(0.5)
				# obs, rew, done, _ = env.step(action)
				radar_obs, speed, rew, done, distance = env.step_straight(
							final_action, 0)

				# Sum all episodic rewards as we go along
				ep_ret += rew
				state_data = []
				state_data.insert(0,speed)
				state_data.insert(1,distance)
				state_data = np.array(state_data)
				state_obs = state_data
		# Track episodic length
		finally:
			env.destroy()
			ep_len = t

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret

def eval_policy(cov_var,cov_mat,policy, env, render=False):
	"""
		The main function to evaluate our policy with. It will iterate a generator object
		"rollout", which will simulate each episode and return the most recent episode's
		length and return. We can then log it right after. And yes, eval_policy will run
		forever until you kill the process. 

		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on
			render - Whether we should render our episodes. False by default.

		Return:
			None

		NOTE: To learn more about generators, look at rollout's function description
	"""
	# Rollout with the policy and environment, and log each episode's data
	for ep_num, (ep_len, ep_ret) in enumerate(rollout(cov_var,cov_mat,policy, env, render)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)

def choose_action_straight( choice):
	# Choice will be a continuous value between 0-1

	action = []
	action = [float(choice), 0, 0, False]

	return action

