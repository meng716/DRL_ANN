import os

import pandas as pd

from DQN import *
from algo import *

from visualization_manager import VisualizationManager

# set seed
seed = 10

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

if __name__ == "__main__":
    data_dir = './data'  # specify relative path to data directory (e.g., './data', not './data/variant_0')

    variant = 0  # specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)

    # initialize environment
    no_of_states = 24  # length of observation
    no_of_actions = 5
    cnn_state = 4
    load_model = 1

    agent_type = "DQN"
    model_type = "nn"
    obs_type = "nn3"
    visualization_mode = {"training": False, "validation": False}
    run_mode = {
        "training": False,
        "validation": True,
        "testing": True
    }
    no_training_episodes = 3000
    no_validation_episodes = 100

    val_hist = []
    test_hist = []

    """ Specify which improvement technology(s) is to be used for training. Apply multiple techniques at once 
    by concatenating the strings. Example: double dqn + prioritized replay  -> ddpr 
    dd: double dqn
    pr: prioritized replay
    es: error scaling
    sg: gamma schedule
    tm1, tm2: target model schedule 
   """
    techs = [""]

    """ This allows to tune multiple hyperparameters at one run by looping the list.
        The parameter effected should be manually changed accordingly.
        Hyperparameter used for this placeholder can be: no_of_layer, replay_buffer_size, epsilon, learning_rate, etc..
        Could also serve as placeholder for loading different models.  
   """

    # Models with highest validation score for each variant are as below
    hypers = ["saved_models/best/Variant0_nn3_['dd_prdd']_dd.h5",
              "saved_models/best/Variant1_nn3_['tm2_more_epoch']_tm2.h5",
              "saved_models/best/Variant_v2_['baseline']_from_static.h5"
              ]

    """
    Train various variants with its buffer size
    """
    var = [0, 1, 2]
    memory_size = [10000, 100000, 30000]

    for j in range(len(var)):
        for i in range(1):
            variant = var[j]
            analysis_label = ""
            training_label = ""

            if load_model == 0:
                model_weight_path = f"./saved_models/Variant{variant}/Variant{variant}_{obs_type}" \
                                    f"_{training_label}.h5"
            else:
                model_weight_path = hypers[j]

            env = Environment(variant, data_dir)
            env.set_model_type(model_type)
            env.set_get_obs(obs_type)
            visualization_manager = VisualizationManager(env, no_training_episodes, no_validation_episodes,
                                                         mode_selection=visualization_mode)
            """ Create animation for specific episodes 
                by default visualizationManager create the first, middle and last episode of training/validation history.
                if visualization for specific episodes are needed, fill in and uncomment following code block. 
            """
            validation_episode_list = [0]
            visualization_manager.set_validation_episode_list(validation_episode_list)
            #

            """ 
            Specify DQN hyperparameters here:  
            """
            gamma_DQN = 0.95
            epsilon_DQN = 0.1
            batch_size_DQN = 128
            target_model_time_DQN = 20
            alpha_DQN = 0.000008
            no_layers = 5
            load_model_DQN = load_model

            # nn_DQN

            no_layer_DQN, units_DQN, output_size_DQN, activation_DQN, load_DQN = \
                (no_layers, 128, no_of_actions, tf.nn.relu, load_model_DQN)

            #### Hyperparameter tuning stops here, DO NOT change the code below #####
            if agent_type == "DQN":
                agent = DQN_Agent(no_of_states, no_of_actions, gamma_DQN, epsilon_DQN, batch_size_DQN,
                                  target_model_time_DQN,
                                  alpha_DQN, visualization_manager, variant, load_model_DQN, training_label)
                agent.set_model_type(model_type)
                agent.set_memory(memory_size[variant])

                if run_mode["training"] and techs[0] != "":
                    if "dd" in techs[i]:
                        agent.set_double_DQN(True)
                    if "pr" in techs[i]:
                        agent.set_prioriotized_replay(True, memory_size[variant])
                    if "es" in techs[i]:
                        agent.set_error_scaling(True)
                    if "sg" in techs[i]:
                        agent.set_gamma_schedule(True)
                    if "tm1" in techs[i]:
                        agent.set_target_model_time_schedule(1)
                    if "tm2" in techs[i]:
                        agent.set_target_model_time_schedule(2)

                # DQN_set model
                if model_type == "nn":
                    mode = agent.nn_model(no_layer_DQN, units_DQN, output_size_DQN,
                                          activation_DQN, load_DQN)
                    target_model = agent.nn_model(no_layer_DQN, units_DQN, output_size_DQN,
                                                  activation_DQN, load_DQN)

                else:
                    raise RuntimeError(f"model type {model_type} is not supported!")

                agent.set_model(mode, target_model)

                agent.set_model_weight_path(model_weight_path)
                agent.load_model_weight()


            else:
                raise RuntimeError(f"agent type {agent_type} is not supported!")

            visualization_manager.set_agent(agent.name)

            visualization_manager.sanity_check(no_training_episodes, agent.actors)

            # Start training
            if run_mode["training"]:
                print(f"double_dqn: {agent.doubleDQN}, prioritized: {agent.prioritized_replay}"
                      f", ES: {agent.error_scaling}, memory_size : {memory_size}"
                      f"load_model: {load_model}, model_path: {model_weight_path}")
                rew, total_episodes = agent.train(env, no_training_episodes)
                print(f"{agent.name}_{agent.model_type} training score: {np.mean(rew)}")

                # # # CREATE ANIMATION
                visualization_manager.prepare_data_for_ani()
                visualization_manager.create_env_animation(save_path=f"./outputs/Variant{variant}", save_pngs=False,
                                                           fps=5)

            # Start Validating
            if run_mode["validation"]:
                rew, _ = agent.run(env, no_validation_episodes, 'validation')
                print(f"{agent.name}_{agent.model_type} validation score: {np.mean(rew)}")

                """ 
                Uncomment this part to store movement statics for policy analysis
                """
                # pd.DataFrame(env.obs_list).to_csv(f"./logs/movement/Variant{variant}/obv_list_{analysis_label}")
                # pd.DataFrame(env.act_list).to_csv(f"./logs/movement/Variant{variant}/act_list_{analysis_label}")
                # pd.DataFrame(rew).to_csv(f"./logs/movement/Variant{variant}/rew_list_{analysis_label}")
                # pd.DataFrame(agent.reward_list).to_csv(f"./logs/movement/Variant{variant}/step_reward_{analysis_label}")

                # # CREATE ANIMATION
                visualization_manager.prepare_data_for_ani()
                visualization_manager.create_env_animation(save_path=f"./outputs/Variant{variant}", save_pngs=False,
                                                           fps=5)

                val_hist.append(np.mean(rew))

            # Start testing
            if run_mode["testing"]:
                rew, _ = agent.run(env, no_validation_episodes, 'testing')
                print(f"{agent.name}_{agent.model_type} testing score: {np.mean(rew)}")

                test_hist.append(np.mean(rew))

        print(val_hist)
        print(test_hist)
