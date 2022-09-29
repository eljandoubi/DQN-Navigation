
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Agent"

[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png

# DQN-Navigation
This repository contains material from Udacity's [Value-based Methods](https://github.com/udacity/Value-based-methods) github.


## Introduction

For this project, we train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, an agent must get an average score of +13 over 100 consecutive episodes.


For the Optional Challenge: Learning from Pixels


This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view of the environment.

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.9.

	- __Linux__ or __Mac__: 
	```bash 
    conda create --name drlnd 
    source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd 
	activate drlnd
	```
2. Follow the instructions in [Pytorch](https://pytorch.org/) web page to install pytorch and its dependencies (PIL, numpy,...). For Windows and cuda 11.6

    ```bash
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
    ```
	

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

    ```bash
    pip install gym[box2d]
    ```
    
4. Follow the instructions in [Navigation](https://github.com/udacity/Value-based-methods/tree/main/p1_navigation) to get the environment.
	
5. Clone the repository, and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/eljandoubi/DQN-Navigation.git
cd DQN-Navigation/python
pip install .
```

6. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

7. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

## Training and inference
You can train and/or inference Navigation (Pixels) environment:

First, go to `p1_navigation/`.

Then, run the traing and/or inference cell of `Deep_Q_Network_Navigation(_Pixels).ipynb`.

The pre-trained model with the highest score is stored in `Navigation_(Pixels_)checkpoint`.


## Implementation and Resultats

The implementation and resultats are discussed in the report.
