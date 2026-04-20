import pytest
import jax
import jax.numpy as jnp
from scirex.operators.training.trainer_jax import Trainer, SimpleDataLoader
from scirex.operators.layers.channel_mlp_jax import ChannelMLP
from scirex.operators.training.adamw_jax import AdamW


def test_trainer_init():
    """Test Trainer initialization."""
    model = ChannelMLP(in_channels=2, out_channels=1, n_layers=1)
    trainer = Trainer(model=model, n_epochs=1, verbose=False)

    assert trainer.model is model
    assert trainer.n_epochs == 1


def test_trainer_train_step():
    """Test a single training step."""
    # Create simple model
    model = ChannelMLP(in_channels=2, out_channels=1, n_layers=1)
    x = jnp.ones((1, 2, 4, 4))
    key = jax.random.PRNGKey(0)
    params = model.init(key, x)

    # Create dummy data loader
    batch = {"x": x, "y": jnp.ones((1, 1, 4, 4))}
    train_loader = SimpleDataLoader([batch])

    # Create optimizer
    optimizer = AdamW(lr=0.01)

    # Create trainer
    trainer = Trainer(model=model, n_epochs=1, verbose=False, params=params)

    # Mock scheduler (just return lr)
    def scheduler(step):
        return 0.01

    # This should run without errors
    try:
        print("Starting training...")
        trainer.train(train_loader, {}, optimizer, scheduler)
        print("Training completed successfully")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Training failed with error: {e}")


def test_simple_data_loader():
    """Test SimpleDataLoader."""
    batches = [{"x": jnp.ones((1, 2))}, {"x": jnp.ones((1, 2))}]
    loader = SimpleDataLoader(batches)

    assert len(loader) == 2
    assert loader.dataset.__len__() == 2

    # Test iteration
    count = 0
    for batch in loader:
        assert "x" in batch
        count += 1
    assert count == 2