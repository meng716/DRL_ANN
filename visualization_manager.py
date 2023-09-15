import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import os
from environment import Environment


class VisualizationManager:
    def __init__(
            self,
            env: Environment,
            no_training_episodes: int,
            no_validation_episodes: int,
            mode_selection: dict = {"training": True, "validation": True},
            training_actor_list: list = [0],
    ) -> None:
        '''
        Handles the data storage and post-processing for PPO and DQN. (SAC to be updated)
        PPO:    any number of actors and episodes.
        DQN:    always assumed to have 1 actor for simpler data storage.
        '''
        self.mode_selection = mode_selection
        self.training_episode_list = [0, int(no_training_episodes / 2), no_training_episodes - 1]
        self.validation_episode_list = [0, int(no_validation_episodes / 2), no_validation_episodes - 1]
        self.training_actor_list = training_actor_list
        self.agent_type = None
        self.current_time = None  # TO READ FROM AGENT LATER
        self.target_loc = env.target_loc
        self.max_time_steps = env.episode_steps
        self.max_response_time = env.max_response_time
        self.eligible_cells = env.eligible_cells

        self.plot_episode_list = []
        self.plot_actor_list = []
        self.episode_count = 0
        self.actor_count = 0
        self.is_record = False
        self.is_mode_match = False
        self.current_mode = None

    def set_agent(self, agent):
        self.agent_type = agent

    def set_training_episode_list(self, training_episode_list):
        self.training_episode_list = training_episode_list

    def set_validation_episode_list(self, validation_episode_list):
        self.validation_episode_list = validation_episode_list

    def reset_mode(self, mode):
        '''
        Restore the visualization manager to initial state.
        Prepare for next stage (training or validation).
        '''

        self.plot_episode_list = []
        self.plot_actor_list = []
        self.episode_count = 0
        self.actor_count = 0
        self.is_record = False
        self.is_mode_match = False

        if mode == "training" and self.mode_selection["training"]:
            self.is_mode_match = True
            self.current_mode = "training"
            self.plot_episode_list = self.training_episode_list
            self.plot_actor_list = self.training_actor_list

        elif mode == "validation" and self.mode_selection["validation"]:
            self.is_mode_match = True
            self.current_mode = "validation"
            self.plot_episode_list = self.validation_episode_list
            self.plot_actor_list = [0]

        self.state_buffer = np.zeros(
            (len(self.plot_episode_list), len(self.plot_actor_list), self.max_time_steps + 1, 28),
            dtype=np.float32)
        self.reward_buffer = np.zeros((len(self.plot_episode_list), len(self.plot_actor_list), self.max_time_steps + 1),
                                      dtype=np.float32)

    def record_data_for_ani(self, state, reward, time):
        if self.is_record:
            self.state_buffer[self.episode_count, self.actor_count, time, :] = state
            self.reward_buffer[self.episode_count, self.actor_count, time] = reward

    def check_is_record(self, episode, actor=0):
        self.is_record = False

        if self.is_mode_match and self.current_mode == "training":
            self.is_record = episode in self.training_episode_list and actor in self.training_actor_list

        elif self.is_mode_match and self.current_mode == "validation":
            self.is_record = episode in self.validation_episode_list

    def increment_actor_count(self):
        if self.is_record:
            self.actor_count += 1

    def increment_episode_count(self):
        if self.is_record:
            self.episode_count += 1

    def reset_actor_count(self):
        self.actor_count = 0

    def reset_episode_count(self):
        self.episode_count = 0

    def prepare_data_for_ani(self):

        env_state = self.state_buffer[..., :-3].reshape(self.state_buffer.shape[0],
                                                        self.state_buffer.shape[1],
                                                        self.state_buffer.shape[2], 5, 5)

        # CREATE NON-ELIGIBLE CELLS LIST
        non_eligible_cells = []
        for i in range(env_state.shape[3]):
            for j in range(env_state.shape[4]):
                if (i, j) not in self.eligible_cells:
                    non_eligible_cells.append((i, j))
                    env_state[..., i, j] = np.NaN  # MAKE THE CELL WHITE AND TRANSPARENT

        agent_loc = self.state_buffer[..., -3:-1]
        agent_load = self.state_buffer[..., -1].astype(int)

        env_grid = np.meshgrid(np.linspace(0, 4, 5), np.linspace(0, 4, 5), indexing="xy")

        cumulative_reward = self.reward_buffer.cumsum(axis=2)

        self.env_state = env_state
        self.agent_loc = agent_loc
        self.agent_load = agent_load
        self.env_grid = env_grid
        self.reward = cumulative_reward
        self.non_eligible_cells = non_eligible_cells

    def create_dir(self, path):
        try:
            os.mkdir(path)
        except OSError:
            pass
            # print(f"Warning: {path} already exists.")

    def create_env_animation(
            self,
            save_path: str = None,
            save_pngs: bool = False,
            fps: int = 5,
    ) -> None:
        '''
        Create animations for specified episodes (and actos) for the whole training / validation cycle.
        '''

        # BREAK THE FUNCTION IF THERE IS NOTHING TO PLOT
        if self.current_mode is None:
            return None

        # CREATE DIRECTORY FOR AGENT
        if self.agent_type == "PPO":
            save_path = save_path + "/PPO"
        elif self.agent_type == "DQN":
            save_path = save_path + "/DQN"
        self.create_dir(save_path)

        # CREATE DIRECTORY AT RUN TIME OF THE MODEL
        save_path = save_path + "/" + self.current_time
        self.create_dir(save_path)

        # CREATE DIRECTORY FOR RUN CYCLE
        if self.current_mode == "training":
            save_path = save_path + "/training"
        elif self.current_mode == "validation":
            save_path = save_path + "/validation"
        self.create_dir(save_path)

        # CREATE PLOTS
        for episode_idx, episode in enumerate(self.plot_episode_list):
            print(f"Creating animation for Episode {episode:02d}.")

            total_time_steps = self.env_state.shape[2]

            episode_path = save_path + f"/episode_{episode:02d}"
            self.create_dir(episode_path)

            for actor_idx, actor in enumerate(self.plot_actor_list):

                path = episode_path

                if self.agent_type == "PPO":
                    path += f"/actor_{actor:02d}"
                    self.create_dir(path)

                # PLOT THE STATE AT FIRST TIMESTEP
                fig, ax, pm, sc, ti = self.create_env_plot(
                    env_state=self.env_state[episode_idx, actor_idx, ...],
                    env_grid=self.env_grid,
                    non_eligible_cells=self.non_eligible_cells,
                    agent_loc=self.agent_loc[episode_idx, actor_idx, ...],
                    agent_load=self.agent_load[episode_idx, actor_idx, ...],
                    target_loc=self.target_loc,
                    reward=self.reward[episode_idx, actor_idx, ...],
                    time=0)

                # DEFINE FUNCTION FOR ANIMATION
                def animate(i):
                    pm.set_array(self.env_state[episode_idx, actor_idx, i, ...].flatten())
                    sc.set_offsets((self.agent_loc[episode_idx, actor_idx, i, 1],
                                    self.agent_loc[episode_idx, actor_idx, i, 0]))
                    ti.set_text(
                        f"LOAD = {self.agent_load[episode_idx, actor_idx, i]}, REWARD = {self.reward[episode_idx, actor_idx, i]}")
                    return pm, sc, ti

                # CREATE ANIMATION
                ani = FuncAnimation(
                    fig, animate,
                    frames=total_time_steps,
                    blit=True,
                    repeat=True
                )
                FFwriter = animation.FFMpegWriter(fps=fps)
                ani.save(path + "/animation.mp4", writer=FFwriter)

                plt.clf()
                plt.cla()
                plt.close(fig)
                del fig, ax

                # SAVE STATES TO INDIVIDUAL PICTURES
                if save_pngs:
                    for time in range(total_time_steps):
                        filename = f"image_{time:04d}"

                        fig, ax, pm, sc, ti = self.create_env_plot(
                            env_state=self.env_state[episode_idx, actor_idx, ...],
                            env_grid=self.env_grid,
                            non_eligible_cells=self.non_eligible_cells,
                            agent_loc=self.agent_loc[episode_idx, actor_idx, ...],
                            agent_load=self.agent_load[episode_idx, actor_idx, ...],
                            target_loc=self.target_loc,
                            reward=self.reward[episode_idx, actor_idx, ...],
                            time=time)

                        print("Saving %s" % filename)
                        fig.savefig(path + "/" + filename, bbox_inches="tight")
                        plt.clf()
                        plt.cla()
                        plt.close(fig)
                        del fig, ax
            print(f"Finish creating animation for Episode {episode:02d}.")
        print("CREATE ANIMATION COMPLETED.")

    def create_env_plot(
            self,
            env_state: np.ndarray,
            env_grid: np.ndarray,
            non_eligible_cells: list,
            agent_loc: np.ndarray,
            agent_load: np.ndarray,
            target_loc: np.ndarray,
            reward: np.ndarray,
            time: int = 0):
        '''
        Create plot of one episode for one actor at a certain timestep.
        '''

        fig, ax = plt.subplots()

        # COLORMAP SETTING
        cmap = "OrRd"
        grid_on = True

        # PLOT ENVIRONMENT STATE
        pm = ax.pcolormesh(env_grid[0], env_grid[1], env_state[time, :],
                           cmap=cmap, vmin=0, vmax=self.max_response_time)
        ax.set_xticks(np.arange(1, 5) - 0.5, minor=True)
        ax.set_yticks(np.arange(1, 5) - 0.5, minor=True)
        ax.grid(grid_on, which='minor', axis='both')

        # PLOT TARGET LOCATION
        ax.scatter(target_loc[1], target_loc[0], marker='*', s=500, color='tab:orange')

        # PLOT AGENT LOCATION
        sc = ax.scatter(agent_loc[time, 1], agent_loc[time, 0], marker='o', s=100, color='tab:blue')

        # CROSS OUT NON-ELIGIBLE CELLS
        for (i, j) in non_eligible_cells:
            ax.scatter(j, i, marker='x', s=1500, color='black')

        #
        ti = ax.set_title(f"LOAD = {agent_load[time]}, REWARD = {reward[time]}")
        fig.colorbar(pm, ax=ax)
        ax.invert_yaxis()
        fig.tight_layout()

        return fig, ax, pm, sc, ti

    def sanity_check(self, no_episodes: int, no_actors: int = 1):
        assert self.agent_type in ["PPO", "DQN", "SAC"], f"The agent type {self.agent_type} is not supported."

        if not self.agent_type == "PPO":
            assert self.training_actor_list == [0], f"There should be no actor list for agent type {self.agent_type}"

        for episode in self.training_episode_list:
            assert episode <= no_episodes - 1, \
                f"The requested plot Episode {episode} is out of range. (Index start from 0!)"

        if self.agent_type == "PPO":
            for actor in self.training_actor_list:
                assert actor <= no_actors - 1, \
                    f"The requested plot Actor {actor} is out of range. (Index start from 0!)"

    # Todo: print saving dir and specify whether it is training or validation
