import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import pprint as pp
import gym
import utils
import os

class Actor(object) :
	def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size) :
		self.sess = sess
		self.s_dim = state_dim
		self.a_dim = action_dim
		self.action_bound = action_bound
		self.learning_rate = learning_rate
		self.tau = tau
		self.batch_size = batch_size

		self.inp = tf.placeholder(shape = [None, self.s_dim], dtype = tf.float32)

		self.out, self.scaled_out = self.create_actor_network('main_actor')

		self.network_params = tf.trainable_variables()

		self.target_out, self.target_scaled_out = self.create_actor_network('target_actor')

		self.target_network_params = tf.trainable_variables()[
			len(self.network_params):]

		
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
												  tf.multiply(self.target_network_params[i], 1. - self.tau))
				for i in range(len(self.target_network_params))]

				

	def create_actor_network(self, scope, reuse = False) :
		with tf.variable_scope(scope, reuse = reuse) :
			net = self.inp
			net = slim.fully_connected(net, 400, activation_fn = tf.nn.relu)
			net = slim.fully_connected(net, 300, activation_fn = tf.nn.relu)
			net = slim.fully_connected(net, self.a_dim, activation_fn = tf.nn.tanh)
			scaled_out = tf.multiply(net, self.action_bound)
			return net, scaled_out

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)

	def predict(self, inputs):
		return self.sess.run(self.scaled_out, feed_dict={
			self.inp: inputs
		})

	def predict_target(self, inputs):
		return self.sess.run(self.target_scaled_out, feed_dict={
			self.inp: inputs
		})

class Critic(object) :

	def __init__(self, sess, state_dim, action_dim, learning_rate, tau, inp_actions) :

		self.sess = sess
		self.s_dim = state_dim
		self.a_dim = action_dim
		self.learning_rate = learning_rate
		self.tau = tau
		self.inp_actions = inp_actions

		self.inp = tf.placeholder(shape = [None, self.s_dim], dtype = tf.float32)
		self.action = tf.placeholder(shape = [None, self.a_dim], dtype = tf.float32)

		self.total_out,_ = self.create_critic_network('main_critic', self.inp_actions)
		self.out1, self.out2 = self.create_critic_network('main_critic', self.action, reuse = True)

		self.target_out1, self.target_out2 = self.create_critic_network('target_critic', self.action)

		self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main_critic')
		self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_critic')
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
												  tf.multiply(self.target_network_params[i], 1. - self.tau))
				for i in range(len(self.target_network_params))]

		self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
		
		self.loss = tf.reduce_mean(tf.square(self.out1 - self.predicted_q_value)) + tf.reduce_mean(tf.square(self.out2 - self.predicted_q_value))
		self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list = self.network_params)


	def create_critic_network(self, scope, actions, reuse = False) :

		with tf.variable_scope(scope, reuse = reuse) :
			net = tf.concat([self.inp, actions], axis = 1)
			net = slim.fully_connected(net, 400)
			net = slim.fully_connected(net, 100)
			net = slim.fully_connected(net, 1, activation_fn = None)
			net1 = net
			net = tf.concat([self.inp, actions], axis = 1)
			net = slim.fully_connected(net, 400)
			net = slim.fully_connected(net, 100)
			net = slim.fully_connected(net, 1, activation_fn = None)
			net2 = net
		return net1, net2	
				


	def update_target_network(self):
		self.sess.run(self.update_target_network_params)
	
	def predict1(self, inputs, action):
		return self.sess.run(self.out1, feed_dict={
			self.inp: inputs,
			self.action: action
		})

	def predict2(self, inputs, action):
		return self.sess.run(self.out2, feed_dict={
			self.inp: inputs,
			self.action: action
		})	

	def predict_target1(self, inputs, action):
		return self.sess.run(self.target_out1, feed_dict={
			self.inp: inputs,
			self.action: action
		})

	def predict_target2(self, inputs, action):
		return self.sess.run(self.target_out2, feed_dict={
			self.inp: inputs,
			self.action: action
		})	



def eval(env, actor) :
	s = env.reset()
	done = False
	episode_r = 0
	while not done :
		env.render()
		action = actor.predict(np.reshape(s, (1, actor.s_dim)))
		s2, r, done, _ = env.step(action)
		s = s2
		episode_r += r

	print('During evaluation the mean episode reward is ', episode_r)	




				

def main(args):

	if not os.path.exists(args['save_dir']) :
		os.makedirs(args['save_dir'])

	with tf.Session() as sess:

		env = gym.make(args['env'])
		np.random.seed(int(args['random_seed']))
		tf.set_random_seed(int(args['random_seed']))
		env.seed(int(args['random_seed']))

		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0]
		action_bound = int(env.action_space.high[0])
		

		actor = Actor(sess, state_dim, action_dim, action_bound,
							 float(args['actor_lr']), float(args['tau']),
							 int(args['minibatch_size']))

		critic = Critic(sess, state_dim, action_dim,
							   float(args['critic_lr']), float(args['tau']),
							   actor.scaled_out)
							   
		
		saver = tf.train.Saver()
		saver.restore(sess, os.path.join(args['save_dir'], args['env']))
		eval(env, actor)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='provide arguments for TD3 agent')

	# agent parameters
	parser.add_argument('--actor-lr', help='actor network learning rate', default=0.001)
	parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
	parser.add_argument("--start_timesteps", default=1e4, type=int)
	parser.add_argument('--tau', help='soft target update parameter', default=0.005)
	parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=100)
	parser.add_argument("--policy_noise", help = 'std of noise added ', default=0.2, type=float)		
	parser.add_argument("--noise_clip", default=0.5, type=float)		
	parser.add_argument("--discount", default=0.99, type=float)	
	parser.add_argument("--policy_freq", default=2, type=int)
	parser.add_argument("--expl_noise", default=0.1, type=float)

	# run parameters
	parser.add_argument('--env', help='choose the gym env', default='HalfCheetah-v1')
	parser.add_argument('--random-seed', help='random seed for repeatability', default=0)
	parser.add_argument("--max_timesteps", default=1e6, type=float)	
	parser.add_argument("--eval_episodes", default=100, type=float)
	parser.add_argument("--save_dir", default='./models/', help = 'save directory')	
	parser.add_argument("--save_timesteps", default=2e5, type=float)	
		
	
	args = vars(parser.parse_args())
	
	pp.pprint(args)

	main(args)




			
	

				

