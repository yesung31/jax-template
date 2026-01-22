import jax.numpy as jnp
import optax

from core.model import Model, ModelOutput
from models.networks.template_network import TemplateNetwork


class TemplateModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = TemplateNetwork()

    def configure_optimizers(self, learning_rate):
        return optax.adam(learning_rate)

    def training_step(self, params, batch):
        x, y = batch
        logits = self.net.apply({"params": params}, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()

        return ModelOutput(loss=loss, extra={"logits": logits}).log(
            "train_loss", loss, prog_bar=True
        )

    def validation_step(self, state, batch):
        # Re-use training logic for forward pass and loss
        x, y = batch
        output = self.training_step(state.params, batch)
        logits = output.extra["logits"]

        # Compute accuracy
        accuracy = jnp.mean(jnp.argmax(logits, -1) == y)

        return output.log("val_loss", output.loss).log("val_acc", accuracy, prog_bar=True)
