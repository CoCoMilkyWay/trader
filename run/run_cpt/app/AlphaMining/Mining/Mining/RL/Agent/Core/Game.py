import numpy as np
        
from abc import ABC, abstractmethod

import gymnasium as gym

class AbstractGame(ABC):
    """
    wrapper class of environment as game(used by MuZero framework)
    """

    @abstractmethod
    def __init__(self, seed=None):
        pass

    @abstractmethod
    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        pass

    def player_id(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return 0

    @abstractmethod
    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        pass

    def close(self):
        """
        Properly close the game.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Display the game observation.
        """
        pass

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the action to play for the player {self.player_id()}: ")
        while int(choice) not in self.legal_actions(): # type: ignore
            choice = input("Illegal action. Enter another action : ")
        return int(choice)

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        raise NotImplementedError

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return str(action_number)

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, env:gym.Env):
        self.env = env
        self.info = {}

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, terminated, truncated, self.info = self.env.step(action)
        
        # Reshape it to 3D (1, 1, n)
        observation_3d = np.array(observation, dtype=np.uint8).reshape(1, 1, -1)
        return observation_3d, reward, terminated

    def player_id(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return 0

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        action_masks = self.info['action_masks']
        legal_actions = np.where(action_masks)[0].tolist()
        return legal_actions

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        observation, self.info = self.env.reset()
        # Reshape it to 3D (1, 1, n)
        observation_3d = np.array(observation, dtype=np.uint8).reshape(1, 1, -1)
        return observation_3d

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        # valid = False
        # while not valid:
        #     valid, action = self.env.human_input_to_action()
        # return action
        pass

    def action_to_string(self, action):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        # return self.env.action_to_human_input(action)
        pass


import numpy

class Game_CartPole(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self):
        self.env = gym.make("CartPole-v1")

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, terminated, truncated, _ = self.env.step(action)
        return numpy.array([[observation]]), reward, terminated

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(2))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return numpy.array([[self.env.reset()]])

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Push cart to the left",
            1: "Push cart to the right",
        }
        return f"{action_number}. {actions[action_number]}"
