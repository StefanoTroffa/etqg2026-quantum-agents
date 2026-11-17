from abc import ABC, abstractmethod
import numpy as np

class QuantumGameEnv(ABC):
    """
    Abstract Base Class for Quantum Game Environments.
    Defines the standard interface required by training pipelines and MCTS.
    """
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Resets the environment and returns the initial state."""
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Returns the current state representation (numpy array)."""
        pass

    @abstractmethod
    def get_valid_actions(self) -> list:
        """Returns a list of valid action indices."""
        pass

    @abstractmethod
    def step(self, action: int, return_state: bool = True) -> tuple:
        """
        Executes the action.
        Returns:
            If return_state is True: (next_state, reward, done, info)
            Else: (None, reward, done, info)
        """
        pass

    @abstractmethod
    def clone(self):
        """Creates a fast copy of the environment state for tree searches."""
        pass

    @property
    @abstractmethod
    def current_player(self) -> int:
        """Returns the player whose turn it is (typically 1 or -1)."""
        pass

    @current_player.setter
    @abstractmethod
    def current_player(self, val: int):
        pass

    @property
    @abstractmethod
    def winner(self) -> int:
        """Returns the winner (1, -1) or None if no winner yet or draw."""
        pass

    @property
    @abstractmethod
    def done(self) -> bool:
        """Returns True if the game is over, False otherwise."""
        pass
