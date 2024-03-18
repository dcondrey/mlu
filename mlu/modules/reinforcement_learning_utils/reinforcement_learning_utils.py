import numpy as np
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Environment:
    def __init__(self):
        self.state = None
        self.is_done = False

    def reset(self):
        """
        Initialize or reset the environment state.
        """
        self.state = self._get_initial_state()
        self.is_done = False
        return self.state

    def step(self, action):
        """
        Apply an action and return the new state, reward, and whether the episode has ended.

        Parameters:
        - action: The action to apply.

        Returns:
        - newState: The new state after the action.
        - reward: The reward received after applying the action.
        - is_done: Whether the episode has ended.
        """
        newState, reward, self.is_done = self._apply_action(action)
        return newState, reward, self.is_done

    def _get_initial_state(self):
        """
        Define how to get the initial state of the environment.
        """
        raise NotImplementedError("This method needs to be implemented by subclasses.")

    def _apply_action(self, action):
        """
        Define how an action affects the environment.
        """
        raise NotImplementedError("This method needs to be implemented by subclasses.")

def train_model(environment, model, episodes):
    """
    Train a model on the environment for a specified number of episodes.

    Parameters:
    - environment: An instance of the Environment class or its subclass.
    - model: The model to be trained.
    - episodes (int): The number of episodes to train the model.
    """
    for episode in range(episodes):
        state = environment.reset()
        total_reward = 0
        while not environment.is_done:
            action = model.predict(state)
            newState, reward, done = environment.step(action)
            model.update(state, action, reward, newState, done)
            state = newState
            total_reward += reward
        logging.info(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")