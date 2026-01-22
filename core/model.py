from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
from flax import struct
from flax.training import train_state


@struct.dataclass
class ModelOutput:
    """Standardized output for model steps with a fluent logging API."""

    loss: Any
    metrics: dict[str, Any] = struct.field(default_factory=dict)
    pbar: dict[str, Any] = struct.field(default_factory=dict)
    extra: dict[str, Any] = struct.field(default_factory=dict)

    def log(self, name: str, value: Any, prog_bar: bool = False):
        """
        Logs a metric.

        Args:
            name: Name of the metric.
            value: Value to log.
            prog_bar: Whether to also display this metric in the progress bar.
        """
        new_metrics = {**self.metrics, name: value}
        new_pbar = {**self.pbar, name: value} if prog_bar else self.pbar
        return self.replace(metrics=new_metrics, pbar=new_pbar)


class Model(ABC):
    """Abstract base class for models."""

    def __init__(self, **kwargs):
        self.net = None  # Subclasses must initialize this

    @abstractmethod
    def configure_optimizers(self, learning_rate):
        """Returns the optimizer (Optax gradient transformation)."""
        pass

    def create_train_state(self, rng, input_shape, learning_rate):
        """Creates and returns the initial TrainState."""
        if self.net is None:
            raise ValueError("self.net must be initialized in __init__")

        params = self.net.init(rng, jnp.ones(input_shape))["params"]
        tx = self.configure_optimizers(learning_rate)
        return train_state.TrainState.create(apply_fn=self.net.apply, params=params, tx=tx)

    @abstractmethod
    def training_step(self, params, batch) -> ModelOutput:
        """
        Computes loss and auxiliary metrics for training.
        """
        pass

    @abstractmethod
    def validation_step(self, state, batch) -> ModelOutput:
        """
        Performs a single evaluation step.
        """
        pass

    def test_step(self, state, batch) -> ModelOutput:
        """
        Performs a single test step. Defaults to validation_step.
        """
        return self.validation_step(state, batch)
