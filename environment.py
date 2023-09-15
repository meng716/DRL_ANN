# actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

# positions in grid:
# - (0,0) is upper left corner
# - first index is vertical (increasing from top to bottom)
# - second index is horizontal (increasing from left to right)

# if new item appears in a cell into which the agent moves/at which the agent stays in the same time step,
# it is not picked up (if agent wants to pick it up, it has to stay in the cell in the next time step)

import random
import pandas as pd
from copy import deepcopy
from itertools import compress
import numpy as np


class Environment(object):
    def __init__(self, variant, data_dir):
        self.obs_type = None
        self.variant = variant
        self.vertical_cell_count = 5
        self.horizontal_cell_count = 5
        self.vertical_idx_target = 2
        self.horizontal_idx_target = 0
        self.target_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.episode_steps = 200
        self.max_response_time = 15 if self.variant == 2 else 10
        self.reward = 25 if self.variant == 2 else 15
        self.data_dir = data_dir
        self.model_type = None

        self.training_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/training_episodes.csv')
        self.training_episodes = self.training_episodes.training_episodes.tolist()
        self.validation_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/validation_episodes.csv')
        self.validation_episodes = self.validation_episodes.validation_episodes.tolist()
        self.test_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/test_episodes.csv')
        self.test_episodes = self.test_episodes.test_episodes.tolist()

        self.remaining_training_episodes = deepcopy(self.training_episodes)
        self.validation_episode_counter = 0

        # add new fields
        self.item_times = []
        self.item_locs = []
        self.step_count = 0
        self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.agent_load = 0  # number of items loaded (0 or 1, except for first extension, where it can be 0,1,2,3)
        self.opt_loc = self.agent_loc
        self.opt_true_reward = 0

        if self.variant == 0 or self.variant == 2:
            self.agent_capacity = 1
        else:
            self.agent_capacity = 3

        if self.variant == 0 or self.variant == 1:
            self.max_dist = 10
            self.eligible_cells = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                                   (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                                   (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
                                   (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
                                   (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
        else:
            self.max_dist = 14
            self.eligible_cells = [(0, 0), (0, 2), (0, 3), (0, 4),
                                   (1, 0), (1, 2), (1, 4),
                                   (2, 0), (2, 2), (2, 4),
                                   (3, 0), (3, 1), (3, 2), (3, 4),
                                   (4, 0), (4, 1), (4, 2), (4, 4)]

        self.plot_buffer = np.zeros(28, )
        self.obs_list = []
        self.act_list =[]

    def set_model_type(self, model_type: str):
        self.model_type = model_type

    def set_get_obs(self, obs_type):
        self.obs_type = obs_type

    # initialize a new episode (specify if training, validation, or testing via the mode argument)
    def reset(self, mode):
        modes = ['training', 'validation', 'testing']
        if mode not in modes:
            raise ValueError('Invalid mode. Expected one of: %s' % modes)

        self.step_count = 0
        self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.agent_load = 0  # number of items loaded (0 or 1, except for first extension, where it can be 0,1,2,3)
        self.item_locs = []
        self.item_times = []

        if mode == "testing":
            episode = self.test_episodes[0]
            self.test_episodes.remove(episode)
        elif mode == "validation":
            episode = self.validation_episodes[self.validation_episode_counter]
            self.validation_episode_counter = (self.validation_episode_counter + 1) % 100
        else:
            if not self.remaining_training_episodes:
                self.remaining_training_episodes = deepcopy(self.training_episodes)
            episode = random.choice(self.remaining_training_episodes)
            self.remaining_training_episodes.remove(episode)
        self.data = pd.read_csv(self.data_dir + f'/variant_{self.variant}/episode_data/episode_{episode:03d}.csv',
                                index_col=0)
        # SAVE STATE FOR ANIMATION
        self.save_plot_buffer()

        return self.get_obs()

    def reset_greedy(self, mode):
        modes = ['training', 'validation', 'testing']
        if mode not in modes:
            raise ValueError('Invalid mode. Expected one of: %s' % modes)

        self.step_count = 0
        self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.agent_load = 0  # number of items loaded (0 or 1, except for first extension, where it can be 0,1,2,3)
        self.item_locs = []
        self.item_times = []

        if mode == "testing":
            episode = self.test_episodes[0]
            self.test_episodes.remove(episode)
        elif mode == "validation":
            episode = self.validation_episodes[self.validation_episode_counter]
            self.validation_episode_counter = (self.validation_episode_counter + 1) % 100
        else:
            if not self.remaining_training_episodes:
                self.remaining_training_episodes = deepcopy(self.training_episodes)
            episode = random.choice(self.remaining_training_episodes)
            self.remaining_training_episodes.remove(episode)
        self.data = pd.read_csv(self.data_dir + f'/variant_{self.variant}/episode_data/episode_{episode:03d}.csv',
                                index_col=0)

        # SAVE STATE FOR ANIMATION
        self.save_plot_buffer()

        # return self.get_obs()

    # take one environment step based on the action act
    def step(self, act):
        self.step_count += 1

        rew = 0

        # done signal (1 if episode ends, 0 if not)
        if self.step_count == self.episode_steps:
            done = 1
        else:
            done = 0

        # agent movement
        if act != 0:
            if act == 1:  # up
                new_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
            elif act == 2:  # right
                new_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
            elif act == 3:  # down
                new_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
            elif act == 4:  # left
                new_loc = (self.agent_loc[0], self.agent_loc[1] - 1)

            # print(f"act: {act}, loc: ({self.agent_loc[0]}, {self.agent_loc[1]}))")
            if new_loc in self.eligible_cells:
                self.agent_loc = new_loc
                rew += -1

        # item pick-up
        if (self.agent_load < self.agent_capacity) and (self.agent_loc in self.item_locs):
            self.agent_load += 1
            idx = self.item_locs.index(self.agent_loc)
            self.item_locs.pop(idx)
            self.item_times.pop(idx)
            rew += self.reward / 2

        # item drop-off
        if self.agent_loc == self.target_loc:
            rew += self.agent_load * self.reward / 2
            self.agent_load = 0

        # track how long ago items appeared
        self.item_times = [i + 1 for i in self.item_times]

        # remove items for which max response time is reached
        mask = [i < self.max_response_time for i in self.item_times]
        self.item_locs = list(compress(self.item_locs, mask))
        self.item_times = list(compress(self.item_times, mask))

        # add items which appear in the current time step
        new_items = self.data[self.data.step == self.step_count]
        new_items = list(zip(new_items.vertical_idx, new_items.horizontal_idx))
        new_items = [i for i in new_items if i not in self.item_locs]  # not more than one item per cell
        self.item_locs += new_items
        self.item_times += [0] * len(new_items)

        self.update_opt_dest()
        # get new observation
        next_obs = self.get_obs()

        # SAVE STATE FOR ANIMATION
        self.save_plot_buffer()
        return rew, next_obs, done


    @staticmethod
    def manhattan_dist(o1, o2):
        return np.abs(o1[0] - o2[0]) + np.abs(o1[1] - o2[1])

    def dist(self, o1, o2):
        if self.variant != 2:
            return self.manhattan_dist(o1, o2)
        else:
            if (self.is_m_zone(o1) & self.is_m_zone(o2)) | (o1[1] == o2[1]):
                return self.manhattan_dist(o1, o2)
            else:
                if self.is_m_zone(o1):
                    mid = self.min_m_dist(o2)
                    return self.manhattan_dist(mid, o2) + self.dist(mid, o1)
                else:
                    mid = self.min_m_dist(o1)
                    return self.manhattan_dist(mid, o1) + self.dist(mid, o2)

    @staticmethod
    def min_m_dist(o1):
        if o1[1] == 0:
            return 3, 0
        else:
            return 0, 4

    @staticmethod
    def is_m_zone(loc):
        if loc[1] == 0:
            return loc[0] > 2
        elif loc[1] == 4:
            return loc[0] == 0
        else:
            return True

    def net_reward(self, item_loc, item_times):
        if (self.max_response_time - item_times) < self.dist(self.agent_loc, item_loc):
            return 0
        return self.reward - (self.dist(self.agent_loc, item_loc) + self.dist(self.target_loc, item_loc))


    def opt_destination(self):
        dest = self.target_loc
        if len(self.item_locs) == 0:
            return self.agent_loc
        elif self.agent_load / self.agent_capacity == 1:
            return dest
        else:
            max_reward = 0
            for i in range(len(self.item_locs)):
                temp = self.net_reward(self.item_locs[i], self.item_times[i])
                if temp > max_reward:
                    max_reward = temp
                    dest = self.item_locs[i]
            return dest



    def opt_list(self):
        result = []

        for i in range(len(self.item_locs)):
            loc = self.item_locs[i]
            time = self.item_times[i]
            net_reward = self.net_reward(loc, time)
            result.append((loc[0], loc[1],
                           self.dist(self.agent_loc, loc),
                           self.dist(loc, self.target_loc),
                           net_reward,
                           time,
                           ))

        if self.agent_load == self.agent_capacity:
            target = self.target_loc
            t = (target[0], target[1],
                 self.dist(self.agent_loc, target),
                 0,
                 8, 8,
                 )

            return np.array([t, t, t, t])
        return sorted(result, key=lambda x: (-x[4]))

    # return the top 3 item location and time that agent should go to

    def opt_list_normalized(self):
        # [(loc[0], loc[1], dist_to_item, item_to_target, net_reward, time, avg_reward)]
        result = []

        for i in range(len(self.item_locs)):
            loc = self.item_locs[i]
            time = self.item_times[i]
            net_reward = self.net_reward(loc, time)
            result.append((loc[0] / 4, loc[1] / 4,
                           np.interp(self.dist(self.agent_loc, loc), [0, self.max_dist], [0, 1]),
                           np.interp(self.dist(loc, self.target_loc), [0, self.max_dist], [0, 1]),
                           np.interp(net_reward, [0, self.reward], [0, 1]),
                           np.interp(time, [1, self.max_response_time], [0, 1]),
                           ))
        #    sort by net reward, and then by time
        if self.agent_load == self.agent_capacity:
            target = self.target_loc
            t = (target[0] / 4, target[1] / 4,
                 np.interp(self.dist(self.agent_loc, target), [0, self.max_dist], [0, 1]),
                 0,
                 0.8, 0.8,
                 )
            return np.array([t, t, t])
        return sorted(result, key=lambda x: (-x[4]))


    def update_opt_dest(self):
        next_opt_dest = self.opt_destination()
        # new Item appears to be the better option, update the opt_loc and init true reward of the new item
        if next_opt_dest == self.agent_loc:
            self.opt_loc = self.agent_loc
            self.opt_true_reward = 0
        elif next_opt_dest != self.target_loc and next_opt_dest != self.opt_loc:
            self.opt_loc = next_opt_dest
            i = self.item_locs.index(next_opt_dest)
            time = self.item_times[i]
            self.opt_true_reward = self.net_reward(next_opt_dest, time)
        else:
            self.opt_loc = next_opt_dest




    def get_obs_nn3(self):
        cand = self.opt_list()
        empty_dummy = (self.agent_loc[0], self.agent_loc[1], 0, 0, 0, 5)

        result = [empty_dummy, empty_dummy, empty_dummy]
        for i in range(min(3, len(cand))):
            result[i] = cand[i]
        result = [e for t in result for e in t]

        obvs = np.array([
            self.target_loc[0], self.target_loc[1],
            self.dist(self.agent_loc, self.target_loc),
            self.agent_loc[0], self.agent_loc[1], self.agent_load,
        ])
        x = np.concatenate((result, obvs))
        return x.reshape(1, len(x))

    def get_obs_nn4(self):
        cand = self.opt_list_normalized()
        empty_dummy = (self.agent_loc[0] / 4, self.agent_loc[1] / 4, 0, 0, 0, 0.5)

        result = [empty_dummy, empty_dummy, empty_dummy]
        for i in range(min(3, len(cand))):
            result[i] = cand[i]
        result = [e for t in result for e in t]

        obvs = np.array([
            self.target_loc[0] / 4, self.target_loc[1] / 4,
            np.interp(self.dist(self.agent_loc, self.target_loc), [0, self.max_dist], [0, 1]),
            self.agent_loc[0] / 4, self.agent_loc[1] / 4, self.agent_load,
        ])
        x = np.concatenate((result, obvs))
        return x.reshape(1, len(x))




    def get_obs(self):
        if self.obs_type == "nn3":  # baseline feature
            return self.get_obs_nn3()
        elif self.obs_type == "nn4": # normalized feature
            return self.get_obs_nn4()
        else:
            return None

    def save_plot_buffer(self):
        '''
        Save the state for animation.
        '''
        plot_buffer = np.zeros((5, 5))

        for idx, item_loc in enumerate(self.item_locs):
            plot_buffer[item_loc[0]][item_loc[1]] = self.max_response_time - self.item_times[idx]

        plot_buffer = plot_buffer.flatten().tolist()

        plot_buffer.extend(self.agent_loc)
        plot_buffer.append(self.agent_load)

        plot_buffer = np.array(plot_buffer)
        plot_buffer = plot_buffer.reshape((1, 28))

        self.plot_buffer = plot_buffer


