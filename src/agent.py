import os
from responses import activate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from dis import disco
import numpy as np
from tensorflow.compat.v1 import variable_scope, Session, placeholder, layers, distributions, train, global_variables_initializer, train
tf.compat.v1.disable_eager_execution()

class Actor(object):
    def __init__(self, sess, a_dim, n_features, action_bound, lr=0.001) -> None:
        self.sess = sess
        self.s = placeholder(tf.float32, [1, n_features], "state")
        self.a = placeholder(tf.float32, shape=[1, a_dim], name="act")
        self.td_error = placeholder(tf.float32, None, name="td_error")

        l1 = layers.dense(
            inputs=self.s,
            units=256,
            activation=tf.nn.relu6,
            kernel_initializer=tf.random_normal_initializer(.0, .1),
            bias_initializer=tf.constant_initializer(0.1),
            name="l1"
        )

        l2 = layers.dense(
            inputs=l1,
            units=256,
            activation=tf.nn.relu6,
            kernel_initializer=tf.random_normal_initializer(.0, .1),
            bias_initializer=tf.constant_initializer(0.1),
            name="l2"
        )

        mu = layers.dense(
            inputs=l2,
            units=a_dim,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(.0, .1),
            bias_initializer=tf.constant_initializer(0.1),
            name="mu"
        )

        sigma = layers.dense(
            inputs=l2,
            units=a_dim,
            activation=tf.nn.softplus,
            kernel_initializer=tf.random_normal_initializer(.0, .1),
            bias_initializer=tf.constant_initializer(1.),
            name="sigma"
        )

        global_step = tf.Variable(0, trainable=False)
        self.mu, self.sigma = tf.squeeze(mu), tf.squeeze(sigma + 0.1)
        self.normal_dist = distributions.Normal(self.mu, self.sigma)
        self.action = tf.clip_by_value(self.normal_dist.sample(), action_bound[0], action_bound[1])

        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)
            self.exp_v = log_prob * self.td_error
            self.exp_v += 0.01 * self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]    
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v
    
    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s: s})

class Critic(object):
    def __init__(self, sess, n_features, discount, lr=0.01) -> None:
        self.sess = sess
        self.discount = discount
        with tf.name_scope('inputs'):
            self.s = placeholder(tf.float32, [1, n_features], "state")
            self.v_ = placeholder(tf.float32, [1, 1], name="v_next")
            self.r = placeholder(tf.float32, name="r")

        with variable_scope('Critic'):
            l1 = layers.dense(
                inputs=self.s,
                units=256,
                activation=tf.nn.relu6,
                kernel_initializer=tf.random_normal_initializer(.0, .1),
                bias_initializer=tf.constant_initializer(0.1),
                name="l1"
            )

            l2 = layers.dense(
                inputs=l1,
                units=256,
                activation=tf.nn.relu6,
                kernel_initializer=tf.random_normal_initializer(.0, .1),
                bias_initializer=tf.constant_initializer(0.1),
                name="l2"
            )

            self.v = layers.dense(
                inputs=l2,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(.0, .1),
                bias_initializer=tf.constant_initializer(0.1),
                name="V"
            )

        with variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + self.discount * self.v_ - self.v)
            self.loss = tf.square(self.td_error)
        
        with variable_scope('train'):
            self.train_op = train.AdamOptimizer(lr).minimize(self.loss)
    
    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})
        return td_error
    

class Agent:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, discount) -> None:
        print("Num GPUs Available: ", len(
            tf.config.experimental.list_physical_devices('GPU')))
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        # self.sess = Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = Session()
        self.actor = Actor(self.sess, a_dim=action_dim, n_features=state_dim, lr=actor_lr, action_bound=[-1, 1])
        self.critic = Critic(self.sess, n_features=state_dim, lr=critic_lr, discount=discount)
        self.sess.run(global_variables_initializer())
        self.saver = train.Saver(max_to_keep=5)

    def get_action(self, state):
        return self.actor.choose_action(state)

    def feedback(self, s, r, s_, actions):
        self.actor.learn(s, actions, self.critic.learn(s, r, s_))
    
    def save(self):
        self.saver.save(self.sess, "models/actor_critic_max_10_task.ckpt")
