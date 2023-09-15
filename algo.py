from abc import ABC, abstractmethod
from environment import Environment
import tensorflow as tf


class Algorithm(ABC):
    @abstractmethod
    def __init__(self):
        self.state_size = None
        self.name = None
        self.model_type = None


    def set_model_type(self, model_type:str):
        self.model_type = model_type

    def train(self, env: Environment, no_training_episodes):
        rew, total_episodes = self.run(env, no_training_episodes, mode='training')
        return rew, total_episodes

    # Define neural network
    def nn_model(self, no_hidden_layer, hidden_node_units, output_size, activation, load_old_model):
        observation_input = tf.keras.Input(shape=(self.state_size,), dtype=tf.float32)
        prev_layer = observation_input
        for i in range(no_hidden_layer):
            next_layer = tf.keras.layers.Dense(units=hidden_node_units, activation=activation)(prev_layer)
            prev_layer = next_layer
        output = tf.keras.layers.Dense(output_size)(prev_layer)

        # actor_critic
        if output_size == 1:
            output = tf.squeeze(output, axis=1)

        nn_model = tf.keras.Model(inputs=observation_input, outputs=output)

        return nn_model


    def set_model(self, model1, model2):
        pass

    @abstractmethod
    def run(self, env, no_episodes, mode="training"):
        pass

