import jax
import jax.numpy as jnp
from flax import linen as nn

class ProjectionLayerDerivative:
    def __init__(self, params, v_L):
        self.params = params
        self.v_L = v_L
        
        projection_params = self.params['projection_layer']

        # Store weights
        self.W1 = projection_params['dense_0']['kernel']
        self.b1 = projection_params['dense_0']['bias']
        self.W2 = projection_params['dense_1']['kernel']
        self.b2 = projection_params['dense_1']['bias']

def first_order_derivative(params, v_L):
    # 1. Extract projection weights directly
    projection_params = params['projection_layer']
    W1 = projection_params['dense_0']['kernel']
    b1 = projection_params['dense_0']['bias']
    W2 = projection_params['dense_1']['kernel']
    
    # 2. Forward pass to hidden layer
    z = jnp.dot(v_L, W1) + b1
    
    # 3. 1st derivative of GELU
    flat_z = z.flatten()
    d_gelu = jax.vmap(jax.grad(nn.gelu))
    g_prime = d_gelu(flat_z).reshape(z.shape)
    
    # 4. Spectral derivatives
    v_prime = exact_spectral_derivative(v_L, deriv_order=1)
    v_prime_x, v_prime_y = v_prime[..., 0], v_prime[..., 1]
    
    # 5. Analytical forward-mode contraction
    # u_x = v'_x * W1
    u_x = jnp.dot(v_prime_x, W1)  
    u_y = jnp.dot(v_prime_y, W1)
    
    # Q'(v_L) * v' = (u_x * g') * W2
    u_prime_x = jnp.dot(u_x * g_prime, W2) 
    u_prime_y = jnp.dot(u_y * g_prime, W2)
    
    return jnp.stack([u_prime_x, u_prime_y], axis=-1)

# NO jax.jit here
def second_order_derivative(params, v_L):
    # 1. Extract projection weights directly
    projection_params = params['projection_layer']
    W1 = projection_params['dense_0']['kernel']
    b1 = projection_params['dense_0']['bias']
    W2 = projection_params['dense_1']['kernel']
    
    # 2. Forward pass to hidden layer
    z = jnp.dot(v_L, W1) + b1
    
    # 3. 1st and 2nd derivatives of GELU
    flat_z = z.flatten()
    d_gelu = jax.vmap(jax.grad(nn.gelu))
    d2_gelu = jax.vmap(jax.grad(jax.grad(nn.gelu)))
    
    g_prime = d_gelu(flat_z).reshape(z.shape)
    g_double_prime = d2_gelu(flat_z).reshape(z.shape)
    
    # 4. Spectral derivatives
    v_prime = exact_spectral_derivative(v_L, deriv_order=1)
    v_prime2 = exact_spectral_derivative(v_L, deriv_order=2)
    
    v_prime_x, v_prime_y = v_prime[..., 0], v_prime[..., 1]
    v_prime2_x, v_prime2_y = v_prime2[..., 0], v_prime2[..., 1]
    
    # -------------------------------------------------------------------
    # Term 1: Q''(v_L) * (v')^2
    # Analytical math: (v'_x W1)^2 * g'' * W2
    # -------------------------------------------------------------------
    u_x = jnp.dot(v_prime_x, W1)
    u_y = jnp.dot(v_prime_y, W1)
    
    term_1_x = jnp.dot((u_x ** 2) * g_double_prime, W2)
    term_1_y = jnp.dot((u_y ** 2) * g_double_prime, W2)
    
    # -------------------------------------------------------------------
    # Term 2: Q'(v_L) * v''
    # Analytical math: (v''_x W1) * g' * W2
    # -------------------------------------------------------------------
    w_x = jnp.dot(v_prime2_x, W1)
    w_y = jnp.dot(v_prime2_y, W1)
    
    term_2_x = jnp.dot(w_x * g_prime, W2)
    term_2_y = jnp.dot(w_y * g_prime, W2)
    
    # -------------------------------------------------------------------
    # Combine and stack
    # -------------------------------------------------------------------
    d2u_dx2 = term_1_x + term_2_x
    d2u_dy2 = term_1_y + term_2_y
    
    return jnp.stack([d2u_dx2, d2u_dy2], axis=-1)