# DRL -refactor log

**Update until 29.05** 
model type (nn, cnn) and its parameters can be specify in main. 
Add model_type in Environment to specify which get_observation to return.

**19.05**

As discussed, some changes are made to the existing code (changes are made on “refactor” branch).
Major changes are as follow:

**Visualization Manager (vm)**
1. Add Environment parameter in Visualization Manager, such that no parameters need to be explicitly assigned when instantiate visualization manager. 
2. By default vm creates animation of the 1st, the middle, and the last episode. Specific episodes can also be created by uncommenting lines in main class.

**Algo**
A new class Algorithm is created. It is the parent class of all algorithms classes (i.e. PPO, SAC, and DQN). Common methods such as train and run (abstract) are moved to the class. I also add a new method to create nn_model. This model allows to specify hyperparameters from numbers of hidden layers to activation method, can also be extended for future needs.

**DQN, PPO, SAC**
1. trainDQN(PPO/SAC) and runDQN(PPO/SAC) methods are refactored according to the changes above. 
2. Structure of neural network can now be specify at __init__

**Main class**
Duplicated codes are removed and refactored.
Steps to train/validate the model
1. Specify no_state_space, no of validation and training episode
2. Specify which agent to run
3. Specify hyper parameters for the agent accordingly
4. Run the code

In general, if the structure of the neural network remains the same, there is no need to make any change in other classes outside of main.

_#TODO:_
1. DQN and PPO save/load weight not implemented.  

