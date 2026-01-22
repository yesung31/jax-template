from abc import ABC


class Callback(ABC):
    """Abstract base class for callbacks."""

    def on_train_start(self, trainer, model):
        """Called when the train begins."""
        pass

    def on_train_end(self, trainer, model):
        """Called when the train ends."""
        pass

    def on_train_epoch_start(self, trainer, model):
        """Called when the epoch begins."""
        pass

    def on_train_epoch_end(self, trainer, model):
        """Called when the epoch ends."""
        pass

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        """Called when the training batch begins."""
        pass

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        """Called when the training batch ends."""
        pass

    def on_validation_start(self, trainer, model):
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self, trainer, model):
        """Called when the validation loop ends."""
        pass

    def on_validation_batch_start(self, trainer, model, batch, batch_idx):
        """Called when the validation batch begins."""
        pass

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx):
        """Called when the validation batch ends."""
        pass

    def on_test_start(self, trainer, model):
        """Called when the test begins."""
        pass

    def on_test_end(self, trainer, model):
        """Called when the test ends."""
        pass

    def on_test_batch_start(self, trainer, model, batch, batch_idx):
        """Called when the test batch begins."""
        pass

    def on_test_batch_end(self, trainer, model, outputs, batch, batch_idx):
        """Called when the test batch ends."""
        pass
