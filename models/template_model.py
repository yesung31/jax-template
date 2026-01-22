import jax.numpy as jnp
import optax
from flax.training import train_state

from models.networks.template_network import TemplateNetwork


class TemplateModel:
    def __init__(self, **kwargs):
        # The model wrapper only holds the network definition.
        # Optimization hyperparameters are passed when creating the state.
        self.net = TemplateNetwork()

    def create_train_state(self, rng, input_shape, learning_rate):
        """Creates initial TrainState with the specified learning rate."""
        params = self.net.init(rng, jnp.ones(input_shape))["params"]
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(apply_fn=self.net.apply, params=params, tx=tx)

    def loss_fn(self, params, batch):
        x, y = batch
        logits = self.net.apply({"params": params}, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
        
        # Metrics to log to WandB/TensorBoard (will be averaged over epoch)
        logs = {"train_loss": loss}
        
        # Metrics to display in the progress bar (current step)
        pbar = {"loss": loss}
        
        return loss, {"logits": logits, "log": logs, "pbar": pbar}

    def eval_step(self, state, batch):
        loss, aux = self.loss_fn(state.params, batch)
        logits = aux["logits"]
        # Compute accuracy
        x, y = batch
        accuracy = jnp.mean(jnp.argmax(logits, -1) == y)
        metrics = {"val_loss": loss, "val_acc": accuracy}
        return metrics
