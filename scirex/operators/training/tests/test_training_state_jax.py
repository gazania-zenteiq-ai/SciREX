import pytest
import tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
from scirex.operators.training.training_state_jax import load_training_state, save_training_state
from scirex.operators.layers.channel_mlp_jax import ChannelMLP
from scirex.operators.training.adamw_jax import AdamW


def test_save_training_state():
    """Test saving training state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        save_name = "test_model"

        # Create dummy model and params
        model = ChannelMLP(in_channels=2, out_channels=1, n_layers=1)
        x = jnp.ones((1, 2, 4, 4))
        key = jax.random.PRNGKey(0)
        params = model.init(key, x)

        # Create dummy optimizer
        optimizer = AdamW(lr=0.01)

        # Save training state
        save_training_state(
            save_dir=save_dir,
            save_name=save_name,
            model=params,
            optimizer=optimizer,
            scheduler=None,
            regularizer=None,
            epoch=1
        )

        # Check that files were created
        assert (save_dir / f"{save_name}_state_dict.pkl").exists()
        assert (save_dir / "optimizer.pkl").exists()
        assert (save_dir / "manifest.pkl").exists()


def test_load_training_state():
    """Test loading training state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        save_name = "test_model"

        # First save
        model = ChannelMLP(in_channels=2, out_channels=1, n_layers=1)
        x = jnp.ones((1, 2, 4, 4))
        key = jax.random.PRNGKey(0)
        params = model.init(key, x)
        optimizer = AdamW(lr=0.01)

        save_training_state(
            save_dir=save_dir,
            save_name=save_name,
            model=params,
            optimizer=optimizer,
            scheduler=None,
            regularizer=None,
            epoch=1
        )

        # Now load
        loaded_params, loaded_optimizer, loaded_scheduler, loaded_regularizer, loaded_epoch = load_training_state(
            save_dir=save_dir,
            save_name=save_name,
            model=params,
            optimizer=optimizer
        )

        assert loaded_params is not None
        # Check that loaded params match
        assert jnp.allclose(loaded_params["params"]["fc_0"]["kernel"], params["params"]["fc_0"]["kernel"])


def test_load_training_state_missing_file():
    """Test loading from non-existent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        save_name = "nonexistent"

        # Should not raise an error, just return None values
        model, optimizer, scheduler, regularizer, epoch = load_training_state(
            save_dir=save_dir, save_name=save_name
        )
        assert model is None
        assert optimizer is None
        assert scheduler is None
        assert regularizer is None
        assert epoch is None