from tensorflow.python.keras.backend import set_session
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

class Agent:
    def __init__(self, mode, model_path, state_dim, action_dim, actor_lr, critic_lr, discount) -> None:
        self.num_hidden_units = 256
        self.model_path = model_path
        self.discount = discount
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.sess = tf.compat.v1.Session()
        self.graph = tf.compat.v1.get_default_graph()
        if mode == "train":
            self.actor, self.critic, self.policy = self.build_network()
        else:
            set_session(self.sess)
            self.actor = load_model(model_path + "actor.save", compile=False)
            self.critic = load_model(model_path + "critic.save", compile=False)
            self.policy = load_model(model_path + "policy.save", compile=False)
            pass

    def build_network(self):
        input = layers.Input(shape=(self.state_dim, ))
        delta = layers.Input(shape=[1])
        dense1 = layers.Dense(self.num_hidden_units, activation="relu")(input)
        dense2 = layers.Dense(self.num_hidden_units, activation="relu")(dense1)
        probs = layers.Dense(self.action_dim, activation="softmax")(dense2)
        values = layers.Dense(1, activation="linear")(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true * K.log(out)
            return K.sum(-log_lik * delta)

        actor = Model(inputs=[input, delta], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.actor_lr), loss=custom_loss)
        critic = Model(inputs=[input], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.critic_lr), loss="mse")
        policy = Model(inputs=[input], outputs=[probs])
        return actor, critic, policy

    def get_action(self, observation):
        state = observation[np.newaxis, :]
        with self.graph.as_default():
            set_session(self.sess)
            return self.policy.predict(state)[0]

    def learn(self, state, action, reward, state_):
        state = state[np.newaxis, :]
        state_ = state_[np.newaxis, :]
        with self.graph.as_default():
            critic_value_ = self.critic.predict(state_)
            critic_value = self.critic.predict(state)

        target = reward + self.discount * critic_value_
        delta = target - critic_value

        actions = np.zeros([1, self.action_dim])
        actions[0][action] = 1.0
        with self.graph.as_default():
            self.actor.fit([state, delta], actions, verbose=0)
            self.critic.fit(state, target, verbose=0)
    
    def save(self):
        with self.graph.as_default():
            self.actor.save(self.model_path + "actor.save")
            self.critic.save(self.model_path + "critic.save")
            self.policy.save(self.model_path + "policy.save")

