import tflearn
import os, sys
import tensorflow as tf
import numpy as np
import random
from pathlib import Path
from parts_from_ddpg import ReplayBuffer

class DefaultActor(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, k, initial_variance):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.variance = np.array([[1]]) # Only for default singlehead. TODO: make this a **kwars??
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.k = 1
        self.initial_variance = initial_variance

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def set_tau(self, tau):
        self.tau = tau

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Here possibly dropout
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights default to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-self.initial_variance, maxval=self.initial_variance)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        
        # Scale output to -action_bound to action_bound
        num_heads = int(self.a_dim/len(self.action_bound[0].flatten())) # misleading because of init
        bounds = np.tile(np.abs(self.action_bound).max(axis=0), num_heads)
        min = np.tile(self.action_bound[0], num_heads)
        max = np.tile(self.action_bound[1], num_heads)
        scaled_out = tf.clip_by_value(tf.multiply(out, bounds), min, max)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def policy(self, inputs):
        a = self.predict(inputs)
        return a, self.variance

    def policy_target(self, inputs, void):
        return self.predict_target(inputs)

    def change_current_head(self):
        pass

    def get_current_head(self):
        return 0

    def save(self, filename):
        tf.saved_model.simple_save(self.sess,
                    "./oracles/"+filename,
                    inputs={"state": self.inputs},
                    outputs={"out": self.scaled_out})
        return



class MultiHeadActor(DefaultActor):
    """Inherits fron ActorNetwork. Last layer is duplicated k times, so it will output k action suggestions. 
    There is a policy() method that outputs the mean action and std"""
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, initial_variance, k):
        super().__init__(sess, state_dim, action_dim*k, action_bound, learning_rate, tau, batch_size, k, initial_variance)
        
        # Because base class beliefs a_dim = actual_a_dim*k:
        self.a_dim = action_dim
        self.k = k

    def policy(self, inputs):
        # multihead is stacked columnwise
        a = np.array(np.hsplit(self.predict(inputs), self.k))
        unscal = np.array(np.hsplit(self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        }), self.k))
        return a.mean(0), np.array(np.cov(unscal.squeeze().T)).reshape(self.a_dim, -1)

    def policy_target(self, inputs):
        a = np.array(np.hsplit(self.predict_target(inputs), self.k))
        return a.mean(0)

class MultiHeadActorTarget(MultiHeadActor):
    """Like MultiHeadActor (average action), but with the variance from the target network."""
    def policy(self, inputs):
        a = np.array(np.hsplit(self.predict(inputs), self.k))
        unscal = np.array(np.hsplit(self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        }), self.k))
        return a.mean(0), np.array(np.cov(unscal.squeeze().T)).reshape(self.a_dim, -1)


class MultiHeadActorNoAverage(MultiHeadActor):
    """This Multihead actor outputs a policy that corresponds to a single head. 
    The actor.current_head value can be randomly set,
     e.g. every episode by calling actor.change_current_head()"""
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, initial_variance, k):
        super().__init__(sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, initial_variance, k)
        self.current_head = 0
        self.change_current_head()

    def policy(self, inputs):
        scaled = np.array(np.hsplit(self.predict(inputs), self.k))
        # TODO: more efficient if the tensor is just copied without a run?
        unscal = np.array(np.hsplit(self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        }), self.k))
        return scaled[self.current_head,:,:], np.array(np.cov(unscal.squeeze().T)).reshape(self.a_dim, -1)

    def policy_target(self, inputs, j_batch):
        return np.array(np.hsplit(self.predict_target(inputs), self.k))[j_batch ,range(inputs.shape[0]),:]

    def change_current_head(self):
        self.current_head = np.random.random_integers(self.k) - 1

    def get_current_head(self):
        return self.current_head

class KHeadActor(MultiHeadActorNoAverage):
    """Inherits from MultiHeadActorNoAverage but the variance estimates stem from the target network"""
    def policy(self, inputs):
        scaled = np.array(np.hsplit(self.predict(inputs), self.k))
        # TODO: more efficient if the tensor is just copied without a run?
        unscal = np.array(np.hsplit(self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        }), self.k))
        return np.array(scaled[self.current_head,:,:]), np.array(np.cov(unscal.squeeze().T)).reshape(self.a_dim, -1)

class Predictor(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, state_dim, action_dim, action_bound, learning_rate, batch_size):
        # self.sess = tf.Session()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.sess.graph.as_default():
                # Actor Network
            self.inputs, self.out, self.scaled_out = self.create_actor_network()

            self.network_params = tf.trainable_variables()

            # This gradient will be provided by the critic network
            self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

            # Combine the gradients here
            self.unnormalized_actor_gradients = tf.gradients(
                self.scaled_out, self.network_params, -self.action_gradient)
            self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

            # Optimization Op
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
                apply_gradients(zip(self.actor_gradients, self.network_params))

            self.num_trainable_vars = len(self.network_params)

            self.sess.run(tf.global_variables_initializer())

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


    def __del__(self):
        self.sess.close()


class ActionBuffer(ReplayBuffer):
    def add(self, s, a):
        experience = (s, a)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])

        return s_batch, a_batch

class Selector(object):
    """Select an action, kalman style"""
    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset

    def select(self, policy, policy_cov, feedback, feedback_cov):
        C = np.eye(policy.shape[1]) # A mapping from feedback to action space 
        gain = policy_cov@(C.T)@np.linalg.inv(C@policy_cov@(C.T) + feedback_cov)*self.scale + self.offset
        action = policy + feedback@gain.T
        # coviance = policy_cov - gain@C@policy_cov
        return action, gain