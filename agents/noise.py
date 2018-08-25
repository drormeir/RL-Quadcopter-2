import numpy as np
import copy
np.random.seed(1)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu = 0.0, theta = 0.15, sigma = 0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        # maximal noise scale is 5.0
        # minimal noise scale is 0.005
        self.__min_theta  = 0.005*theta
        self.__max_theta  = min(0.5,5*theta)
        self.__min_sigma  = 0.005*sigma
        self.__max_sigma  = 5*sigma
        self.__init_theta = theta
        self.__init_sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
            
    def multiply(self, factor):
        self.sigma = np.clip(factor*self.sigma, self.__min_sigma, self.__max_sigma)
        self.theta = np.clip(factor*self.theta, self.__min_theta, self.__max_theta)
        return self.calc_scale()
    
    def calc_scale(self):
        f_theta = 1. if abs(self.theta-self.__init_theta) < 1e-8 else self.theta/self.__init_theta
        f_sigma = 1. if abs(self.sigma-self.__init_sigma) < 1e-8 else self.sigma/self.__init_sigma
        return 0.5*(f_theta + f_sigma)

    def reset_scale(self):
        self.theta = self.__init_theta
        self.sigma = self.__init_sigma
        