from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class Model(ABC):
    """Abstract base class for models."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def create_train_state(self, rng, input_shape, learning_rate):
        """Creates and returns the initial TrainState."""
        pass

    @abstractmethod
    def loss_fn(self, params, batch) -> Tuple[Any, Dict[str, Any]]:
        """
        Computes loss and auxiliary metrics.
        
        Args:
            params: Model parameters.
            batch: Batch of data.
            
        Returns:
            loss: Scalar loss value.
            aux: Dictionary containing:
                - 'log': Metrics to log to WandB/TensorBoard (averaged over epoch).
                - 'pbar': Metrics to display in the progress bar (current step).
                - Any other intermediate results (e.g., 'logits').
        """
        pass

    @abstractmethod
    def eval_step(self, state, batch) -> Dict[str, Any]:
        """
        Performs a single evaluation step.
        
        Args:
            state: Current TrainState.
            batch: Batch of data.
            
        Returns:
            metrics: Dictionary of metrics (e.g., 'val_loss', 'val_acc').
        """
        pass
