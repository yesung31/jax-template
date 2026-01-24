import flax.linen as nn
import jax.numpy as jnp
import optax

from core.model import Model, ModelOutput


class CNNNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


class CNN(Model):
    def __init__(self, learning_rate=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.net = CNNNetwork()

    def configure_optimizers(self, learning_rate):
        return optax.adam(learning_rate)

    def create_train_state(self, rng, input_shape, learning_rate):
        # Override to ensure learning_rate is passed correctly if needed, 
        # but base class handles it. We just need to ensure input_shape is correct.
        # input_shape will be (batch, 28, 28, 1) or similar.
        return super().create_train_state(rng, input_shape, learning_rate)

    def training_step(self, params, batch):
        if isinstance(batch, dict):
            x = batch["input"]
            y = batch["label"]
        else:
            x, y = batch

        logits = self.net.apply({"params": params}, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()

        # Calculate accuracy for monitoring
        accuracy = jnp.mean(jnp.argmax(logits, -1) == y)

        return ModelOutput(loss=loss, extra={"logits": logits}).log(
            "train_loss", loss, prog_bar=True
        ).log("train_acc", accuracy, prog_bar=True)

    def validation_step(self, state, batch):
        if isinstance(batch, dict):
            x = batch["input"]
            y = batch["label"]
        else:
            x, y = batch

        logits = self.net.apply({"params": state.params}, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()

        # Compute accuracy
        accuracy = jnp.mean(jnp.argmax(logits, -1) == y)

        return ModelOutput(loss=loss).log("val_loss", loss, prog_bar=True).log("val_acc", accuracy, prog_bar=True)
