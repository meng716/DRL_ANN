import datetime
import math
from collections import deque
from replay_buffer import PrioritizedReplayBuffer

import numpy as np

import environment
from environment import *
from algo import *


# Validation score (DQN)
# variant 1: 179.885, variant 2: 308.250, variant 3: 128.285
# Validation score (DQN + CNN)
# variant 1: 208.120, variant 2: 359.840, variant 3: 224.220

# Validation score (Greedy)
# variant 1: 218.850, variant 2: 388.895, variant 3: 252.535


class DQN_Agent(Algorithm):
    def __init__(self, no_of_states, no_of_actions, gamma, epsilon, batch_size, target_model_time,
                 alpha, visualization_manager, var, load_model=0, label=""):

        self.name = "DQN"
        self.model_type = None
        self.model_path = None
        self.training_var = None
        self.decay_rate = 0.995
        self.variant = var
        self.label = label
        self.load_model = load_model

        # Optimization techniques
        self.prioritized_replay = False
        self.doubleDQN = False
        self.error_scaling = False
        self.gamma_schedule = False
        self.target_model_time_schedule = 0


        self.state_size = no_of_states  # Record number of state variables
        self.action_size = no_of_actions  # Record number of actions
        self.actors = 0
        self.reward_list = []

        # Hyperparameters
        self.gamma = gamma  # discount rate on future rewards
        self.epsilon = epsilon  # exploration rate
        self.batch_size = batch_size  # maximum size of the batches sampled from memory
        self.target_model_time = target_model_time  # Every 20 time steps update target model weights

        self.initial_beta = 0.01
        self.final_beta = 1
        self.beta = self.initial_beta
        self.replay_buffer_alpha = 0.1


        self.memory = deque(maxlen=4000)  # Set maximum size of experience replay

        # Define your neural network and optimizers
        self.alpha = alpha  # learning rate
        self.model = None
        self.value_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.alpha)
        self.target_model = None


        self.max_timesteps = 200
        current_time = datetime.datetime.now().strftime('%m%d-%H%M')

        self.training_writer = tf.summary.create_file_writer(
            f"./logs/reward/DQN/variant{self.variant}/{current_time}_a{alpha}_e{epsilon}_g{gamma}_tt{target_model_time}"
            f"_{label}")
        self.loss_writer = tf.summary.create_file_writer(
            f"./logs/loss/DQN/variant{self.variant}/{current_time}_a{alpha}_e{epsilon}_g{gamma}_tt{target_model_time}"
            f"_{label}")

        self.visualization_manager = visualization_manager
        self.visualization_manager.current_time = current_time

    # Return state action values, i.e., Q(s,a = 1 & a = 0)

    @tf.function
    def select_action(self, state):
        return self.model(state)[0]

    def set_double_DQN(self,dd:bool):
        self.doubleDQN = dd

    def set_target_model_time_schedule(self, update_time):
        self.target_model_time_schedule = update_time

    def set_training_var(self, var):
        self.training_var = var

    def set_prioriotized_replay(self, pr:bool, size = 4000):
        self.prioritized_replay = pr
        self.memory = PrioritizedReplayBuffer(size, alpha=self.replay_buffer_alpha)


    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_decay_rate(self, decay):
        self.decay_rate = decay

    def set_error_scaling(self, es:bool):
        self.error_scaling= es

    def set_model(self, model: tf.keras.Model, model2: tf.keras.Model):
        self.model = model
        self.target_model = model2

    def set_model_weight_path(self, path):
        self.model_path = path

    def load_model_weight(self):
        if self.load_model == 1:
            self.model.load_weights(self.model_path)
            self.target_model.load_weights(self.model_path)

    def set_memory(self, size):
        self.memory = deque(maxlen=size)

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_gamma_schedule(self, gs):
        self.gamma_schedule = gs

    def set_target_model(self, target_model):
        self.target_model = target_model

    # Save transitions in experience replay
    def record(self, state, action, reward, next_state, done):
        self.memory.append(list((state, action, reward, next_state, done)))

    def normalized_reward(self, reward):
        reward_buffer = []
        if len(self.memory) > 0:
            for i in range(len(self.memory)):
                reward_buffer.append(self.memory[i][2])
        reward_buffer.append(reward)
        reward_buffer_mean = np.mean(reward_buffer)
        reward_buffer_sd = np.std(reward_buffer) + 1e-8
        normalized_reward = (reward_buffer[:-1] - reward_buffer_mean) / reward_buffer_sd
        if len(self.memory) > 0:
            for i in range(len(self.memory)):
                self.memory[i][5] = normalized_reward[i]
        return reward_buffer_mean, reward_buffer_sd

    # Prepare data from experience replay for usage in updates of weights
    def prepare_data(self, minibatch):
        action_buffer = np.zeros((self.batch_size, 2), dtype=np.int32)

        if self.model_type == "nn":
            state_buffer = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
            next_state_buffer = np.zeros((self.batch_size, self.state_size), dtype=np.float32)

        else:
            state_buffer = np.zeros((self.batch_size, 5, 5, 4), dtype=np.float32)
            next_state_buffer = np.zeros((self.batch_size, 5, 5, 4), dtype=np.float32)

        done_buffer = np.zeros((self.batch_size, 1), dtype=np.float32)
        reward_buffer = np.zeros((self.batch_size, 1), dtype=np.float32)

        i = 0
        for x in minibatch:
            state_buffer[i] = x[0]
            next_state_buffer[i] = x[3]
            done_buffer[i] = float(x[4])
            reward_buffer[i] = x[2]
            action_buffer[i] = np.array([i, x[1]], dtype=np.int32)
            i = i + 1
        return state_buffer, next_state_buffer, done_buffer, reward_buffer, action_buffer

    # Perform DQN update of weights w
    @tf.function
    def update_weights_prioritised_replay(self, state_buffer, next_state_buffer, done_buffer, reward_buffer,
                                          action_buffer, weight, gamma):
        # Calculate targets, use tf.math.reduce_prod to have vectors instead of matrices
        target_buffer = (tf.math.reduce_prod(reward_buffer, axis=1) + tf.cast(gamma, tf.float32) * tf.math.reduce_max(
            self.target_model(next_state_buffer), axis=1) * tf.math.reduce_prod(1 - done_buffer, axis=1))

        # double DQN
        if self.doubleDQN:
            next_actions = tf.math.argmax(self.model(next_state_buffer), axis=1)
            next_state_q_values_target = self.target_model(next_state_buffer)
            target_buffer = (tf.math.reduce_prod(reward_buffer, axis=1) + self.gamma * tf.math.reduce_sum(
                tf.one_hot(next_actions, self.action_size) * next_state_q_values_target, axis=1) * tf.math.reduce_prod(
                1 - done_buffer, axis=1))

        # Write down the loss function, i.e., target - predicted values
        with tf.GradientTape() as tape:
            predicted_Q_values = self.model(state_buffer)
            predicted_Q_value_for_action = tf.gather_nd(predicted_Q_values, indices=action_buffer)
            td_error = (target_buffer - predicted_Q_value_for_action) ** 2
            value_loss = tf.reduce_mean(td_error * tf.cast(weight, tf.float32))

        # Update weights
        value_grads = tape.gradient(value_loss, self.model.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.model.trainable_variables))
        return value_loss, td_error

    @tf.function
    def update_weights(self, state_buffer, next_state_buffer, done_buffer, reward_buffer, action_buffer, gamma):
        # Calculate targets, use tf.math.reduce_prod to have vectors instead of matrices
        target_buffer = (tf.math.reduce_prod(reward_buffer, axis=1) + tf.cast(gamma, tf.float32) * tf.math.reduce_max(
            self.target_model(next_state_buffer), axis=1) * tf.math.reduce_prod(1 - done_buffer, axis=1))


        if self.doubleDQN:
            next_actions = tf.math.argmax(self.model(next_state_buffer), axis=1)
            next_state_q_values_target = self.target_model(next_state_buffer)
            target_buffer = (tf.math.reduce_prod(reward_buffer, axis=1) + self.gamma * tf.math.reduce_sum(
                tf.one_hot(next_actions, self.action_size) * next_state_q_values_target, axis=1) * tf.math.reduce_prod(
                1 - done_buffer, axis=1))

        # Write down the loss function, i.e., target - predicted values
        with tf.GradientTape() as tape:
            predicted_Q_values = self.model(state_buffer)
            predicted_Q_value_for_action = tf.gather_nd(predicted_Q_values, indices=action_buffer)
            if self.error_scaling:
                td_error = target_buffer - predicted_Q_value_for_action
                sigma = tf.math.reduce_std(td_error)
                value_loss = tf.reduce_mean(td_error ** 2) / sigma
            else:
                value_loss = tf.reduce_mean((target_buffer - predicted_Q_value_for_action) ** 2)
                td_error = (target_buffer - predicted_Q_value_for_action) **2

        # Update weights
        value_grads = tape.gradient(value_loss, self.model.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.model.trainable_variables))
        return value_loss, td_error

    def save_model(self, filename):
        self.model.save(f"./saved_models/DQN/{filename}")

    def run(self, env: Environment, no_episodes, mode="Training"):
        rew_hist = []
        loss = []

        self.visualization_manager.reset_mode(mode)

        target_model_update_counter = 0  # Initialize counter for target network synchronization
        max_valid_mean = - math.inf
        # Main loop - see DQN pseudo-code in lecture
        for episode in range(no_episodes):

            # Initialize state and rewards for episode
            state = env.reset(mode)
            total_reward = 0

            self.visualization_manager.check_is_record(episode)
            self.visualization_manager.record_data_for_ani(env.plot_buffer, reward=0, time=0)

            # Go through all time steps
            for t in range(self.max_timesteps):
                # Epsilon greedy action selection
                if mode == 'training' and np.random.rand() <= self.epsilon:
                    action = random.randrange(self.action_size)
                else:
                    if self.model_type == "nn":
                        action = np.argmax(self.select_action(state))
                        env.obs_list.append(state.reshape(self.state_size))
                        env.act_list.append(action)
                    else:
                        action = np.argmax(self.select_action(np.expand_dims(state, 0)))

                # Execute the selected action and observe next state + rewards
                reward, next_state, done = env.step(action)
                self.reward_list.append(reward)

                self.visualization_manager.record_data_for_ani(env.plot_buffer, reward, t + 1)

                if mode == 'training':
                    # Record the results of the step
                    if isinstance(self.memory, PrioritizedReplayBuffer):
                        self.memory.add(state, action, reward, next_state, done)
                    else:
                        self.record(state, action, reward, next_state, done)

                    # Perform synchronization of target network every certain no. of steps
                    if target_model_update_counter >= self.target_model_time and np.mod(target_model_update_counter,
                                                                                        self.target_model_time) == 0:
                        w = self.model.get_weights()
                        self.target_model.set_weights(w)

                    # Update weights based on sampled transitions of experience replay
                    if not (len(self.memory) < self.batch_size):
                        if self.prioritized_replay:
                            minibatch, weights, idxes = self.memory.sample(self.batch_size, self.beta)
                            state_buffer, next_state_buffer, done_buffer, reward_buffer, action_buffer = self.prepare_data(
                                minibatch)
                            gamma = self.gamma
                            loss_temp, td_error = self.update_weights_prioritised_replay(state_buffer,
                                                                                         next_state_buffer,
                                                                                         done_buffer,
                                                                                         reward_buffer,
                                                                                         action_buffer,
                                                                                         weights, gamma)
                            new_priorities = np.abs(td_error) + 0.00000001
                            self.memory.update_priorities(idxes, new_priorities)

                        else:
                            minibatch = random.sample(self.memory, self.batch_size)
                            state_buffer, next_state_buffer, done_buffer, reward_buffer, action_buffer = self.prepare_data(
                                minibatch)
                            gamma = self.gamma
                            loss_temp, td_error = self.update_weights(state_buffer, next_state_buffer,
                                                                      done_buffer, reward_buffer,
                                                                      action_buffer, gamma)

                        loss.append(loss_temp)
                        with self.loss_writer.as_default():
                            tf.summary.scalar("loss",loss_temp,step = episode * 200 + t)


                total_reward += reward  # Accumulate reward for this episode
                state = next_state  # Overwrite old state with new state
                target_model_update_counter += 1  # Increment target model update counter

                # End episode if terminal state was reached
                if done:
                    break

            self.visualization_manager.increment_episode_count()

            if mode == 'training' and self.epsilon > 0.01:
                self.epsilon = self.epsilon * self.decay_rate

            if self.target_model_time_schedule == 1:
                if episode == 100:
                    self.target_model_time = 5
                if episode == 200:
                    self.target_model_time = 10

                if episode == 500:
                    self.target_model_time = 20
            elif self.target_model_time_schedule == 2:
                if episode == 0:
                    self.target_model_time = 5

                if episode == 300:
                    self.target_model_time = 10

                if episode == 800:
                    self.target_model_time = 20

                if episode == 1000:
                    self.target_model_time = 40
            else:
                None

            # Test performance of your trained agent every 25 epsiodes
            print("episode: {}/{} | score: {} | e: {:.4f}".format(
                episode + 1, no_episodes, total_reward, self.epsilon))
            if mode == 'training' and self.gamma_schedule and (episode == 2 or episode == 5):
                if episode == 200:
                    self.set_gamma(0.97)
                    print(self.gamma)
                if episode == 400:
                    self.set_gamma(0.99)
                    print(self.gamma)

                if episode > 0 and np.mod(episode + 1, 25) == 0:
                    print(f"intermediary reward after episode {episode + 1}: {np.mean(rew_hist[episode - 25:])}")

            if mode == 'training' and np.mod(episode + 1, 25) == 0:
                rew_valid, tmp = self.run(env, 100, "validation")
                rew_valid_mean = np.mean(rew_valid)
                if episode > no_episodes / 2 and np.mean(rew_valid) > max_valid_mean:
                    max_valid_mean = rew_valid_mean
                    self.model.save_weights(self.model_path, overwrite=True)

                # rew_hist.append(rew_valid_mean)
                with self.training_writer.as_default():
                    tf.summary.scalar('reward', np.mean(rew_valid_mean), step=episode)

                print('intermediary reward after ', episode + 1, ' episodes is ', rew_valid_mean)
                print(f"max intermediary reward is {max_valid_mean}.")

            rew_hist.append(total_reward)


        return rew_hist, loss
