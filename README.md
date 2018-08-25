## Udacity Deep Learning Nanodegree project no. 5:
Teaching a Quadcopter How to Fly using Deep Reinforcement Learning with Keras
======================================================

I was given a quad-copter physical flight simulator. I had to build a Neural Network that controls the velocity of the 4 rotors of the quad-copter, so it will arraive to its designated target on time without crashing on the ground.

In order to do so, I implemented the Actor-Critic Reinforcement Learning algorithm, I also developed a reward function that takes into consideration the current state of the drone (position, velocity, angular velocity) and convert it to a grade that will be maximized in the reinforcement learning process.
The main notebook file for this project is: Quadcopter_Project.ipynb

Additional source code file can be found in task.py, physics_sim_util.py and under the /agents directory

Quadcopter_Project.html is a replica of Quadcopter_Project.ipynb made for offline viewing purpose.

The rest of this readme file was orignated in Udacity and describes how to install and start developing this project.

*********************************************

# Deep RL Quadcopter Controller

*Teach a Quadcopter How to Fly!*

In this project, you will design an agent to fly a quadcopter, and then train it using a reinforcement learning algorithm of your choice! 

## Project Instructions

1. Clone the repository and navigate to the downloaded folder.

```
git clone https://github.com/udacity/RL-Quadcopter-2.git
cd RL-Quadcopter-2
```

2. Create and activate a new environment.

```
conda create -n quadcop python=3.6 matplotlib numpy pandas
source activate quadcop
```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `quadcop` environment. 
```
python -m ipykernel install --user --name quadcop --display-name "quadcop"
```

4. Open the notebook.
```
jupyter notebook Quadcopter_Project.ipynb
```

5. Before running code, change the kernel to match the `quadcop` environment by using the drop-down menu (**Kernel > Change kernel > quadcop**). Then, follow the instructions in the notebook.

6. You will likely need to install more pip packages to complete this project.  Please curate the list of packages needed to run your project in the `requirements.txt` file in the repository.
