from keras import layers, models, optimizers
from keras import backend as K
from agents.replayBuffers import GoodBadReplayBuffer
from agents.noise import OUNoise
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

def QuadcopterDenseLayer(net, n):
    net = layers.Dense(units=n, activation=None)(net)
    #net = layers.Dropout(0.1)(net)
    #net = layers.BatchNormalization()(net)
    net = layers.Activation('elu')(net)
    return net

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size   = state_size
        self.action_size  = action_size
        self.action_low   = action_low
        self.action_high  = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = QuadcopterDenseLayer(states, 32)
        net = QuadcopterDenseLayer(net, 64)
        net = QuadcopterDenseLayer(net, 128)
        net = QuadcopterDenseLayer(net, 32)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
        
    # sugar syntax
    def get_weights(self):
        return self.model.get_weights()
        
    def set_weights(self, w):
        self.model.set_weights(w)
        
    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        return self.model.predict(state)[0]
        
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states  = layers.Input(shape=(self.state_size,),  name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = QuadcopterDenseLayer(states, 32)
        net_states = QuadcopterDenseLayer(net_states, 64)
        net_states = QuadcopterDenseLayer(net_states, 128)

        # Add hidden layer(s) for action pathway
        net_actions = QuadcopterDenseLayer(actions, 32)
        net_actions = QuadcopterDenseLayer(net_actions, 64)
        net_actions = QuadcopterDenseLayer(net_actions, 128)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        # Add more layers to the combined network if needed
        net = QuadcopterDenseLayer(net, 64)
        net = QuadcopterDenseLayer(net, 128)
        net = QuadcopterDenseLayer(net, 32)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net) # , activation='sigmoid'

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
    
    # sugar syntax
    def get_weights(self):
        return self.model.get_weights()
        
    def set_weights(self, w):
        self.model.set_weights(w)
        
class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size  = task.state_size
        self.action_size = task.action_size
        self.action_low  = task.action_low
        self.action_high = task.action_high
        # Noise process
        self.max_unsuccessful_episodes_in_a_row = 10
        self.exploration_mu    = 0
        # theta units are proportional to the action range, so I set it to 1% instead of the original 15%
        self.exploration_theta = 0.01
        self.exploration_sigma = 0.05
        # Replay memory
        self.buffer_size = 10000
        self.batch_size  = 64

        # Algorithm parameters
        self.gamma = 0.99 # discount factor
        self.tau   = 0.1  # for soft update of target parameters
        self.best_learning = -np.inf
        self.reset_learning()
        
        
    def __evaluate(self):
        state = self.task.reset()
        score = 0.
        count = 0
        done = False
        while not done:
            action = self.actor_local.act(state)
            state, reward, done = self.task.step(action)
            score += reward
            count += 1
        score *= self.task.action_repeat * self.task.sim.dt / self.task.sim.runtime
        self.score = score
        self.count = count
        if self.score > self.best_score:
            self.__save_best()
            return True
        return False

    def __save_best(self):
        self.__best_actor_local   = self.actor_local.get_weights()
        self.__best_actor_target  = self.actor_target.get_weights()
        self.__best_critic_local  = self.critic_local.get_weights()
        self.__best_critic_target = self.critic_target.get_weights()
        self.count_unsuccessful_in_a_row = 0
        self.best_score       = self.score
        self.best_score_count = self.count
        if self.best_score > self.best_learning:
            # save best learning
            self.__best_learning_actor_local   = np.copy(self.__best_actor_local)
            self.__best_learning_actor_target  = np.copy(self.__best_actor_target)
            self.__best_learning_critic_local  = np.copy(self.__best_critic_local)
            self.__best_learning_critic_target = np.copy(self.__best_critic_target)
            self.best_learning       = self.best_score
            self.best_learning_count = self.best_score_count
        
    def restore_best(self):
        self.actor_local.set_weights(self.__best_actor_local)
        self.actor_target.set_weights(self.__best_actor_target)
        self.critic_local.set_weights(self.__best_critic_local)
        self.critic_target.set_weights(self.__best_critic_target)
        self.count_unsuccessful_in_a_row = 0
        self.score = self.best_score
        self.count = self.best_score_count
    
    def restore_learning(self):
        self.actor_local.set_weights(self.__best_learning_actor_local)
        self.actor_target.set_weights(self.__best_learning_actor_target)
        self.critic_local.set_weights(self.__best_learning_critic_local)
        self.critic_target.set_weights(self.__best_learning_critic_target)
        self.count_unsuccessful_in_a_row = 0
        self.score = self.best_learning
        self.count = self.best_learning_count
        
    def __update_noise(self):
        if self.__evaluate():
            # improvement in score --> less exploration
            self.noise_scale = self.noise.multiply(0.5)
            self.count_unsuccessful_in_a_row = 0
            return
        self.count_unsuccessful_in_a_row += 1
        if self.count_unsuccessful_in_a_row < self.max_unsuccessful_episodes_in_a_row:
            return
        curr_noise_scale = self.noise_scale
        # increasing the action noise, more exploration
        self.noise_scale = self.noise.multiply(1.5)
        if (self.noise_scale > 1.0) and (self.noise_scale < curr_noise_scale + 1e-6):
            # could not increase action noise
            self.reset_learning()
        else:
            self.restore_best()

    def reset_learning(self):
        # Actor (Policy) Model
        self.actor_local        = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target       = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        # Critic (Value) Model
        self.critic_local       = Critic(self.state_size, self.action_size)
        self.critic_target      = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.set_weights(self.critic_local.get_weights())
        self.actor_target.set_weights(self.actor_local.get_weights())
        self.memory      = GoodBadReplayBuffer(self.buffer_size, self.batch_size)
        self.noise       = OUNoise(self.action_size, mu=self.exploration_mu,\
                                   theta=self.exploration_theta, sigma=self.exploration_sigma)
        self.noise_scale = self.noise.calc_scale()
        self.best_score  = -np.inf
        self.__evaluate()
        return self.reset_episode()
        
    def reset_episode(self, use_noise = True):
        self.noise.reset()
        self.use_noise           = use_noise
        self.last_state          = self.task.reset()
        self.__last_state_reward = self.task.curr_score
        return self.last_state


    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        action = self.actor_local.act(state)
        if self.use_noise:
            ret = list(action + self.noise.sample())  # add some noise for exploration
        else:
            ret = list(action)
        return np.clip(ret, self.task.action_low, self.task.action_high).tolist()

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        if self.use_noise:
            diff_reward = reward - self.__last_state_reward
            self.memory.add(self.last_state, action, diff_reward, reward, next_state, done)
            if self.memory.has_sample():
                # single learn at each step
                self.learn(self.memory.sample())
                if done:
                    self.__update_noise()
                    
        # Roll over last state and action
        self.last_state = next_state
        self.__last_state_reward = reward

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states  = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.task.action_size)
        rewards = np.array([e.step_reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones   = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states    = np.vstack([e.next_state for e in experiences if e is not None])
        # Get predicted next-state actions and Q values from target models
        actions_next   = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),\
                                      (-1, self.task.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights  = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        new_weights    = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        
        