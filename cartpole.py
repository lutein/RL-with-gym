"""
Created on Mon May  8 11:26:55 2017
cartpole control using RL
@author: minty
"""
import argparse
import sys
import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 1000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

class DQN():
  # DQN Agent
  def __init__(self, env):
    # init experience replay
    self.replay_buffer = deque()
    # init some parameters
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.n

    self.create_Q_network()
    self.create_training_method()

    # Init session
    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())
    self.merged=tf.summary.merge_all()
    self.train_writer=tf.summary.FileWriter(FLAGS.log_dir+'/train',self.session.graph)

  def create_Q_network(self):
    # network weights
    with tf.name_scope('weights1'):
      W1 = self.weight_variable([self.state_dim,20])
      tf.summary.histogram('histogram',W1)
    with tf.name_scope('bias1'):
      b1 = self.bias_variable([20])
      tf.summary.histogram('histogram',b1)
    with tf.name_scope('weights2'):
      W2 = self.weight_variable([20,self.action_dim])
      tf.summary.histogram('histogram',W2)
    with tf.name_scope('bias2'):
      b2 = self.bias_variable([self.action_dim])
      tf.summary.histogram('histogram',b2)
    # input layer
    self.state_input = tf.placeholder("float",[None,self.state_dim])
    # hidden layers
    h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)

    self.Q_value = tf.matmul(h_layer,W2) + b2
    tf.summary.histogram('Q_value',self.Q_value)

  def create_training_method(self):
    with tf.name_scope('input'):
      self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
      self.y_input = tf.placeholder("float",[None])#target Q value
    # self.action_input=action_batch,self.Q_value=[float,float],
    Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)

    self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
    tf.summary.scalar('cost function',self.cost)
    self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

  def perceive(self,state,action,reward,next_state,done):
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft()

    if len(self.replay_buffer) > BATCH_SIZE:
      self.train_Q_network()

  def train_Q_network(self):
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    # Step 2: calculate y
    y_batch = []
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
    for i in range(0,BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_batch.append(reward_batch[i])
      else :
        y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
    """if self.time_step%100 == 99:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      summary, _ = self.session.run([self.merged, self.optimizer],
                          feed_dict = {self.state_input:state_batch,
                                       self.y_input:y_batch,
                                       self.action_input:action_batch},
                          options=run_options,
                          run_metadata=run_metadata)
      self.train_writer.add_run_metadata(run_metadata, 'step%03d' % self.time_step)
      self.train_writer.add_summary(summary, self.time_step)
      print('Adding run metadata for', self.time_step)"""
    summary, _, cost_value = self.session.run([self.merged, self.optimizer, self.cost], feed_dict = {self.state_input:state_batch,
                                                                                self.y_input:y_batch,
                                                                                self.action_input:action_batch})
    tf.summary.scalar('cost function value',cost_value)
    if self.time_step%100 == 1:
      self.train_writer.add_summary(summary, self.time_step)


  def egreedy_action(self,state):
    # output of self.Q_value.eval():[[-2.99488783 -0.5567829 ]]
    # Q_value=[-2.99488783 -0.5567829 ]
    Q_value = self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0]
     # produce a float between [0,1] random
    if random.random() <= self.epsilon:
      return random.randint(0,self.action_dim - 1)
    else:
      return np.argmax(Q_value) #return ints

    self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

  def action(self,state):
    #return the action which will have the maximum Q
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0])

  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 1000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

def main(_):
  # initialize OpenAI Gym env and dqn agent
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  env = gym.make(ENV_NAME)
  agent = DQN(env)

  for episode in xrange(EPISODE):
    # initialize task
    state = env.reset()
    # Train
    for step in xrange(STEP):
      action = agent.egreedy_action(state) # e-greedy action for train
      next_state,reward,done,_ = env.step(action)
      # Define reward for agent
      # reward_agent = -1 if done else 0.1
      agent.perceive(state,action,reward,next_state,done)
      state = next_state
      if done:
        break

    # Test every 100 episodes
    if episode % 100 == 0:
      total_reward = 0
      for i in xrange(TEST):
        state = env.reset()
        for j in xrange(STEP):
          env.render()
          action = agent.action(state) # direct action for test
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/TEST
      print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
      if ave_reward >= 200:
        break

  agent.train_writer.close()
  env.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/draft',
                        help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
