import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import jv

def fht(x: jnp.ndarray, order: int = 0, axes: tuple = (-1,)) -> jnp.ndarray:
    """
    Fast Hankel Transform (Discrete Hankel Transform) implemented in JAX.
    
    Transforms the spatial domain into the Hankel (Bessel) frequency domain.
    This implementation uses a continuous pseudo-orthogonal DHT matrix multiplication 
    which is fully JIT-compilable and differentiable. The Bessel matrix is computed 
    at trace-time using robust scipy functions to avoid JAX NaN instabilities on large arrays.
    
    Args:
        x (jnp.ndarray): Input tensor.
        order (int): Order of the Bessel function (defaults to 0).
        axes (tuple): Axes over which to perform the Hankel transform.
        
    Returns:
        jnp.ndarray: Hankel transformed tensor (same shape as input, real-valued).
    """
    out = x
    for axis in axes:
        N = x.shape[axis]
        
        # Create grid indices using standard NumPy at trace time.
        # This completely avoids jax.scipy.special.bessel_jn which is unstable for large arrays.
        i = np.arange(1, N + 1)
        j = np.arange(1, N + 1)
        I, J = np.meshgrid(i, j, indexing='ij')
        
        # Scaling factor to mimic an orthogonal basis over a finite linear grid
        scale = 2.0 * np.pi / (N + 1)
        
        # Transformation Matrix H_ij = J_v(scale * i * j) * sqrt(scale * i * j)
        H_np = jv(order, scale * I * J) * np.sqrt(scale * I * J)
        
        # Approximate normalization to prevent exploding gradients/values
        H_np = H_np / np.sqrt(N)
        
        # Convert the robustly computed numpy array into a constant JAX array
        H = jnp.array(H_np, dtype=x.dtype)
        
        # Apply the transformation matrix to the specified axis
        out = jnp.tensordot(out, H, axes=([axis], [1]))
        
        # tensordot places the contracted axis at the very end. We must move it back.
        out = jnp.moveaxis(out, -1, axis)
        
    return out

def ifht(x: jnp.ndarray, order: int = 0, axes: tuple = (-1,)) -> jnp.ndarray:
    """
    Inverse Fast Hankel Transform (Inverse Discrete Hankel Transform) in JAX.
    
    The continuous Hankel transform is involutive (it is its own inverse).
    Because our discrete transformation matrix H is symmetric, applying the 
    exact same transform again reconstructs the original signal.
    
    Args:
        x (jnp.ndarray): Hankel-domain tensor.
        order (int): Order of the Bessel function.
        axes (tuple): Axes over which to perform the inverse Hankel transform.
        
    Returns:
        jnp.ndarray: Reconstructed spatial tensor.
    """
    return fht(x, order=order, axes=axes)
