import jax.numpy as jnp
import warnings

def __init__(
        self, dim, h=1.0, periodic_in_x=True, periodic_in_y=True, periodic_in_z=True
    ):
        """
        Initialize the FiniteDiff class for computing finite differences.

        See class docstring for detailed parameter descriptions.
        """

        # Check if dim is valid
        if dim not in [1, 2, 3]:
            raise ValueError("dim must be 1, 2, or 3")

        self.dim = dim

        # Set up grid spacing
        if isinstance(h, (int, float)):
            # Create tuple of length dim with repeated h value
            self.h = tuple(h for _ in range(dim))
        else:
            # h is already a tuple/list
            if len(h) != dim:
                raise ValueError(f"For {dim}D, h must be a float or a tuple of length {dim}")
            self.h = tuple(h)  # Convert to tuple

        # Set up periodic conditions
        self.periodic_in_x = periodic_in_x
        if dim >= 2:
            self.periodic_in_y = periodic_in_y
        if dim >= 3:
            self.periodic_in_z = periodic_in_z



def _dx_1st_1d(self,u):
    if self.periodic_in_x:
        # For the case of periodic boundary condition
        dx = (jnp.roll(u,-1, axis=-1)-jnp.roll(u,1,axis=-1))/(2.0*self.h[0])
    else:

        # Predefining the base array
        dx = jnp.zeros_like(u)

        # Now we add values to it which are basically du_dx
        # Interior points: Second-order central differences  (f_{i+1} - f_{i-1})/(2h)
        dx = dx.at[...,1:-1].set((u[...,2:] - u[...,:-2])/(2*self.h[0]))


        # Left boundary: 3rd-order forward differences (-11f_{0} + 18f_{1} - 9f_{2} + 2f_{3})/(6h)
        dx=dx.at[..., 0].set((-11 * u[..., 0] + 18 * u[..., 1] - 9 * u[..., 2] + 2 * u[..., 3]) / (6.0 * self.h[0]))

        # Right boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h)
        dx=dx.at[..., -1].set((-2 * u[..., -4] + 9 * u[..., -3] - 18 * u[..., -2] + 11 * u[..., -1]) / (6.0 * self.h[0]))
    return dx


def _dx_2nd_1d(self, u):

    # For the periodic boundary in x dimension
    if self.periodic_in_x:
        dxx = (jnp.roll(u,-1, axis=-1)-2 * u + jnp.roll(u, 1, axis=-1))/(self.h[0]**2)
        
        #THe non periodic boundary condition
    else:
        # Initialising an empty array
        dxx = jnp.zeros_like(u)

        # Second order central differences in the interior points
        dxx = dxx.at[..., 1:-1].set((u[..., 2:] - 2 * u[..., 1:-1] + u[..., :-2]) / (self.h[0]**2))

        # Second order central difference on the left boundary
        dxx = dxx.at[..., 0].set((2 * u[..., 0] - 5 * u[..., 1] + 4 * u[..., 2] - u[..., 3]) / (self.h[0]**2))

        # Second order central difference on the right boundary
        dxx = dxx.at[..., -1].set((-u[..., -4] + 4 * u[..., -3] - 5 * u[..., -2] + 2 * u[..., -1]) / (self.h[0]**2))


def _dx_1st_2d(self, u):

    if self.periodic_in_x:
        dx = (jnp.roll(u, -1, axis=-2) - jnp.roll(u, 1, axis=-2)) / (2.0 * self.h[0])
    else:
        dx = jnp.zeros_like(u)
        dx = dx.at[..., 1:-1, :].set((u[..., 2:, :] - u[..., :-2, :]) / (2.0 * self.h[0]))
        dx = dx.at[..., 0, :].set((-11 * u[..., 0, :] + 18 * u[..., 1, :] - 9 * u[..., 2, :] + 2 * u[..., 3, :]) / (6.0 * self.h[0]))
        dx = dx.at[..., -1, :].set((-2 * u[..., -4, :] + 9 * u[..., -3, :] - 18 * u[..., -2, :] + 11 * u[..., -1, :]) / (6.0 * self.h[0]))
    return dx


def _dy_1st_2d(self, u):
    if self.periodic_in_y:
        dy = (jnp.roll(u, -1, axis=-1) - jnp.roll(u, 1, axis=-1)) / (2.0 * self.h[1])
    else:
        dy = jnp.zeros_like(u)
        dy = dy.at[..., :, 1:-1].set((u[..., :, 2:] - u[..., :, :-2]) / (2.0 * self.h[1]))
        dy = dy.at[..., :, 0].set((-11 * u[..., :, 0] + 18 * u[..., :, 1] - 9 * u[..., :, 2] + 2 * u[..., :, 3]) / (6.0 * self.h[1]))
        dy = dy.at[..., :, -1].set((-2 * u[..., :, -4] + 9 * u[..., :, -3] - 18 * u[..., :, -2] + 11 * u[..., :, -1]) / (6.0 * self.h[1]))


def _dz_1st_3d(self, u):
    if self.periodic_in_z:
        dz = (jnp.roll(u, -1, axis=-1) - jnp.roll(u, 1, axis=-1)) / (2.0 * self.h[0])
    else:
        # Initialising the array
        dz = jnp.zeros_like(u)

        # Interior point: Second order central difference
        dz = dz.at[..., :, :, 1:-1].set((u[..., :, :, 2:] - u[..., :, :, :-2]) / (2.0 * self.h[2]))

        # Front boundary: Third order central difference
        dz = dz.at[..., :, :, 0].set((-11 * u[..., :, :, 0] + 18 * u[..., :, :, 1] - 9 * u[..., :, :, 2] + 2 * u[..., :, :, 3]) / (6.0 * self.h[2]))

        # Back boundary: Third order central difference
        dz = dz.at[..., :, :, -1].set((-2 * u[..., :, :, -4] + 9 * u[..., :, :, -3] - 18 * u[..., :, :, -2] + 11 * u[..., :, :, -1]) / (6.0 * self.h[2]))
    
    return dz


def _dy_2nd_2d(self, u):
        """Second order derivative with respect to y (2D)."""

        if self.periodic_in_y:
            # Periodic case: use jnp.roll for boundary wrapping
            # Central difference: (f_{i,j+1} - 2f_{i,j} + f_{i,j-1})/(h_{y}²)
            dyy = (jnp.roll(u, -1, axis=-1) - 2 * u + jnp.roll(u, 1, axis=-1)) / (self.h[1] ** 2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dyy = jnp.zeros_like(u)

            # Interior points: Second-order central differences
            # (f_{i,j+1} - 2f_{i,j} + f_{i,j-1})/(h_{y}²)
            dyy = dyy.at[..., :, 1:-1].set((u[..., :, 2:] - 2 * u[..., :, 1:-1] + u[..., :, :-2]) / (self.h[1] ** 2))
            
            # Bottom boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{y}²
            dyy = dyy.at[..., :, 0].set((2 * u[..., :, 0] - 5 * u[..., :, 1] + 4 * u[..., :, 2] - u[..., :, 3]) / (self.h[1] ** 2))
            
            # Top boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{y}²
            dyy = dyy.at[..., :, -1].set((-u[..., :, -4] + 4 * u[..., :, -3] - 5 * u[..., :, -2] + 2 * u[..., :, -1]) / (self.h[1] ** 2))
        
        return dyy


def _dx_2nd_2d(self, u):
        """Second order derivative with respect to x (2D)."""

        if self.periodic_in_x:
            # Periodic case: use jnp.roll for boundary wrapping
            # Central difference: (f_{i+1,j} - 2f_{i,j} + f_{i-1,j})/(h_{x}²)
            dxx = (jnp.roll(u, -1, axis=-2) - 2 * u + jnp.roll(u, 1, axis=-2)) / (self.h[0] ** 2)
        
        else:
            # Non-periodic case: handle boundaries separately
            dxx = jnp.zeros_like(u)

            # Interior points: Second-order central differences
            # (f_{i+1,j} - 2f_{i,j} + f_{i-1,j})/(h_{x}²)
            dxx = dxx.at[..., 1:-1, :].set((u[..., 2:, :] - 2 * u[..., 1:-1, :] + u[..., :-2, :]) / (self.h[0] ** 2))
            
            # Left boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{x}²
            dxx = dxx.at[..., 0, :].set((2 * u[..., 0, :] - 5 * u[..., 1, :] + 4 * u[..., 2, :] - u[..., 3, :]) / (self.h[0] ** 2))
            
            # Right boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{x}²
            dxx = dxx.at[..., -1, :].set((-u[..., -4, :] + 4 * u[..., -3, :] - 5 * u[..., -2, :] + 2 * u[..., -1, :]) / (self.h[0] ** 2))
        
        return dxx


def _dx_1st_3d(self, u):
        """First order derivative with respect to x (3D)."""

        if self.periodic_in_x:
            # Periodic case: use jnp.roll for boundary wrapping
            # Central difference: (f_{i+1,j,k} - f_{i-1,j,k})/(2h_{x})
            dx = (jnp.roll(u, -1, axis=-3) - jnp.roll(u, 1, axis=-3)) / (2.0 * self.h[0])
        
        else:
            # Non-periodic case: handle boundaries separately
            dx = jnp.zeros_like(u)

            # Interior points: Second-order central differences
            # (f_{i+1,j,k} - f_{i-1,j,k})/(2h_{x})
            dx = dx.at[..., 1:-1, :, :].set((u[..., 2:, :, :] - u[..., :-2, :, :]) / (2.0 * self.h[0]))

            # Left boundary: 3rd-order forward differences (-11f_{0} + 18f_{1} - 9f_{2} + 2f_{3})/(6h_{x})
            dx = dx.at[..., 0, :, :].set((-11 * u[..., 0, :, :] + 18 * u[..., 1, :, :] - 9 * u[..., 2, :, :] + 2 * u[..., 3, :, :]) / (6.0 * self.h[0]))

            # Right boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h_{x})
            dx = dx.at[..., -1, :, :].set((-2 * u[..., -4, :, :] + 9 * u[..., -3, :, :] - 18 * u[..., -2, :, :] + 11 * u[..., -1, :, :]) / (6.0 * self.h[0]))
        
        return dx


def _dy_1st_3d(self, u):
    """First order derivative with respect to y (3D)."""

    if self.periodic_in_y:
        # Periodic case: use jnp.roll for boundary wrapping
        # Central difference: (f_{i,j+1,k} - f_{i,j-1,k})/(2h_{y})
        dy = (jnp.roll(u, -1, axis=-2) - jnp.roll(u, 1, axis=-2)) / (2.0 * self.h[1])
    
    else:
        # Non-periodic case: handle boundaries separately
        dy = jnp.zeros_like(u)

        # Interior points: Second-order central differences
        # (f_{i,j+1,k} - f_{i,j-1,k})/(2h_{y})
        dy = dy.at[..., :, 1:-1, :].set((u[..., :, 2:, :] - u[..., :, :-2, :]) / (2.0 * self.h[1]))

        # Bottom boundary: 3rd-order forward differences (-11f_{0} + 18f_{1} - 9f_{2} + 2f_{3})/(6h_{y})
        dy = dy.at[..., :, 0, :].set((-11 * u[..., :, 0, :] + 18 * u[..., :, 1, :] - 9 * u[..., :, 2, :] + 2 * u[..., :, 3, :]) / (6.0 * self.h[1]))

        # Top boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h_{y})
        dy = dy.at[..., :, -1, :].set((-2 * u[..., :, -4, :] + 9 * u[..., :, -3, :] - 18 * u[..., :, -2, :] + 11 * u[..., :, -1, :]) / (6.0 * self.h[1]))
    
    return dy

def _dz_1st_3d(self, u):
        """First order derivative with respect to z (3D)."""

        if self.periodic_in_z:
            # Periodic case: use jnp.roll for boundary wrapping
            # Central difference: (f_{i,j,k+1} - f_{i,j,k-1})/(2h_{z})
            dz = (jnp.roll(u, -1, axis=-1) - jnp.roll(u, 1, axis=-1)) / (2.0 * self.h[2])
        
        else:
            # Non-periodic case: handle boundaries separately
            dz = jnp.zeros_like(u)

            # Interior points: Second-order central differences
            # (f_{i,j,k+1} - f_{i,j,k-1})/(2h_{z})
            dz = dz.at[..., :, :, 1:-1].set((u[..., :, :, 2:] - u[..., :, :, :-2]) / (2.0 * self.h[2]))

            # Front boundary: 3rd-order forward differences (-11f_{0} + 18f_{1} - 9f_{2} + 2f_{3})/(6h_{z})
            dz = dz.at[..., :, :, 0].set((-11 * u[..., :, :, 0] + 18 * u[..., :, :, 1] - 9 * u[..., :, :, 2] + 2 * u[..., :, :, 3]) / (6.0 * self.h[2]))

            # Back boundary: 3rd-order backward differences (-2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1})/(6h_{z})
            dz = dz.at[..., :, :, -1].set((-2 * u[..., :, :, -4] + 9 * u[..., :, :, -3] - 18 * u[..., :, :, -2] + 11 * u[..., :, :, -1]) / (6.0 * self.h[2]))
        
        return dz


def _dx_2nd_3d(self, u):
    """Second order derivative with respect to x (3D)."""

    if self.periodic_in_x:
        # Periodic case: use jnp.roll for boundary wrapping
        # Central difference: (f_{i+1,j,k} - 2f_{i,j,k} + f_{i-1,j,k})/(h_{x}²)
        dxx = (jnp.roll(u, -1, axis=-3) - 2 * u + jnp.roll(u, 1, axis=-3)) / (self.h[0] ** 2)
    
    else:
        # Non-periodic case: handle boundaries separately
        dxx = jnp.zeros_like(u)

        # Interior points: Second-order central differences
        # (f_{i+1,j,k} - 2f_{i,j,k} + f_{i-1,j,k})/(h_{x}²)
        dxx = dxx.at[..., 1:-1, :, :].set((u[..., 2:, :, :] - 2 * u[..., 1:-1, :, :] + u[..., :-2, :, :]) / (self.h[0] ** 2))
        
        # Left boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{x}²
        dxx = dxx.at[..., 0, :, :].set((2 * u[..., 0, :, :] - 5 * u[..., 1, :, :] + 4 * u[..., 2, :, :] - u[..., 3, :, :]) / (self.h[0] ** 2))
        
        # Right boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{x}²
        dxx = dxx.at[..., -1, :, :].set((-u[..., -4, :, :] + 4 * u[..., -3, :, :] - 5 * u[..., -2, :, :] + 2 * u[..., -1, :, :]) / (self.h[0] ** 2))
    
    return dxx

def _dy_2nd_3d(self, u):
    """Second order derivative with respect to y (3D)."""

    if self.periodic_in_y:
        # Periodic case: use jnp.roll for boundary wrapping
        # Central difference: (f_{i,j+1,k} - 2f_{i,j,k} + f_{i,j-1,k})/(h_{y}²)
        dyy = (jnp.roll(u, -1, axis=-2) - 2 * u + jnp.roll(u, 1, axis=-2)) / (self.h[1] ** 2)
    
    else:
        # Non-periodic case: handle boundaries separately
        dyy = jnp.zeros_like(u)

        # Interior points: Second-order central differences
        # (f_{i,j+1,k} - 2f_{i,j,k} + f_{i,j-1,k})/(h_{y}²)
        dyy = dyy.at[..., :, 1:-1, :].set((u[..., :, 2:, :] - 2 * u[..., :, 1:-1, :] + u[..., :, :-2, :]) / (self.h[1] ** 2))
        
        # Bottom boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{y}²
        dyy = dyy.at[..., :, 0, :].set((2 * u[..., :, 0, :] - 5 * u[..., :, 1, :] + 4 * u[..., :, 2, :] - u[..., :, 3, :]) / (self.h[1] ** 2))
        
        # Top boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{y}²
        dyy = dyy.at[..., :, -1, :].set((-u[..., :, -4, :] + 4 * u[..., :, -3, :] - 5 * u[..., :, -2, :] + 2 * u[..., :, -1, :]) / (self.h[1] ** 2))
    
    return dyy

def _dz_2nd_3d(self, u):
    """Second order derivative with respect to z (3D)."""

    if self.periodic_in_z:
        # Periodic case: use jnp.roll for boundary wrapping
        # Central difference: (f_{i,j,k+1} - 2f_{i,j,k} + f_{i,j,k-1})/(h_{z}²)
        dzz = (jnp.roll(u, -1, axis=-1) - 2 * u +
                jnp.roll(u, 1, axis=-1)) / (self.h[2] ** 2)
    
    else:
        # Non-periodic case: handle boundaries separately
        dzz = jnp.zeros_like(u)

        # Interior points: Second-order central differencesr
        # (f_{i,j,k+1} - 2f_{i,j,k} + f_{i,j,k-1})/(h_{z}²)
        dzz = dzz.at[..., :, :, 1:-1].set((u[..., :, :, 2:] - 2 * u[..., :, :, 1:-1] + u[..., :, :, :-2]) / (self.h[2] ** 2))
        
        # Front boundary: 3rd-order forward differences (2f_{0} - 5f_{1} + 4f_{2} - f_{3})/h_{z}²
        dzz = dzz.at[..., :, :, 0].set((2 * u[..., :, :, 0] - 5 * u[..., :, :, 1] + 4 * u[..., :, :, 2] - u[..., :, :, 3]) / (self.h[2] ** 2))
        
        # Back boundary: 3rd-order backward differences (-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h_{z}²
        dzz = dzz.at[..., :, :, -1].set((-u[..., :, :, -4] + 4 * u[..., :, :, -3] - 5 * u[..., :, :, -2] + 2 * u[..., :, :, -1]) / (self.h[2] ** 2))
    
    return dzz

def compute_multiple_derivatives(self, u, derivatives):
    """
    Compute multiple derivatives in a single FFT/IFFT call for better performance.

    Parameters
    ----------
    u : Input tensor.
    derivatives : list
        List of derivative specifications:
        - 1D: list of int (orders)`
        - 2D: list of tuples (order_x, order_y)
        - 3D: list of tuples (order_x, order_y, order_z)

    Returns
    -------
        List of computed derivatives in the same order as derivatives input
    """
    if self.dim == 1:
        return self._compute_multiple_derivatives_1d(u, derivatives)
    elif self.dim == 2:
        return self._compute_multiple_derivatives_2d(u, derivatives)
    elif self.dim == 3:
        return self._compute_multiple_derivatives_3d(u, derivatives)

def derivative(self, u, order):
    """
    Compute Fourier derivative of a given tensor.

    Parameters
    ----------
    u : Input tensor
    order : tuple
        Derivative orders:
        - 1D: (order_x,)
        - 2D: (order_x, order_y)
        - 3D: (order_x, order_y, order_z)

    Returns
    -------
    The derivative of the input tensor
    """
    if len(order) != self.dim:
        raise ValueError(f"For {self.dim}D, order must be a tuple with {self.dim} elements")
        
    if self.dim == 1:
        derivatives = self._compute_multiple_derivatives_1d(u, [order[0]])
    elif self.dim == 2:
        derivatives = self._compute_multiple_derivatives_2d(u, [order])
    elif self.dim == 3:
        derivatives = self._compute_multiple_derivatives_3d(u, [order])

    return derivatives[0]

def partial(self, u, direction="x", order=1):
    """
    Compute partial Fourier derivative along a specific direction.

    Parameters
    ----------
    u : Input tensor
    direction : str, optional
        Direction along which to compute the derivative, by default 'x'
        Options: 'x', 'y' (2D/3D only), 'z' (3D only)
    order : int, optional
        Order of the derivative, by default 1

    Returns
    -------
    The partial derivative of the input tensor
    """
    if direction == "x":
        return self.dx(u, order=order)
    elif direction == "y" and self.dim >= 2:
        return self.dy(u, order=order)
    elif direction == "z" and self.dim >= 3:
        return self.dz(u, order=order)
    else:
        raise ValueError(
            f"Invalid direction '{direction}' for dimension {self.dim}"
        )

def dx(self, u, order=1):
    """Compute derivative with respect to x."""
    if self.dim == 1:
        return self._dx_1d(u, order)
    elif self.dim == 2:
        return self._dx_2d(u, order)
    elif self.dim == 3:
        return self._dx_3d(u, order)

def dy(self, u, order=1):
    """Compute derivative with respect to y (2D/3D only)."""
    if self.dim < 2:
        raise ValueError("dy method only available for 2D and 3D")
    elif self.dim == 2:
        return self._dy_2d(u, order)
    elif self.dim == 3:
        return self._dy_3d(u, order)

def dz(self, u, order=1):
    """Compute derivative with respect to z (3D only)."""
    if self.dim < 3:
        raise ValueError("dz method only available for 3D")
    return self._dz_3d(u, order)

def laplacian(self, u):
    """Compute the Laplacian ∇²f."""
    if self.dim == 1:
        return self.dx(u, order=2)
    elif self.dim == 2:
        return self.dx(u, order=2) + self.dy(u, order=2)
    elif self.dim == 3:
        return self.dx(u, order=2) + self.dy(u, order=2) + self.dz(u, order=2)

def gradient(self, u):
    """Compute the gradient ∇f (returns vector field)."""
    if self.dim == 1:
        return self.dx(u).unsqueeze(-2)
    elif self.dim == 2:
        return jnp.stack([self.dx(u), self.dy(u)], axis=-3)
    elif self.dim == 3:
        return jnp.stack([self.dx(u), self.dy(u), self.dz(u)], axis=-4)

def divergence(self, u):
    """Compute the divergence ∇·u (for vector fields)."""
    expected_dims = {1: 1, 2: 2, 3: 3}
    if u.shape[-self.dim - 1] != expected_dims[self.dim]:
        raise ValueError(
            f"For {self.dim}D, input must have {expected_dims[self.dim]} components in the vector dimension"
        )

    if self.dim == 1:
        return self.dx(u.squeeze(-2))
    elif self.dim == 2:
        return self.dx(u[..., 0, :, :]) + self.dy(u[..., 1, :, :])
    elif self.dim == 3:
        return self.dx(u[..., 0, :, :, :]) + self.dy(u[..., 1, :, :, :]) + self.dz(u[..., 2, :, :, :])

def curl(self, u):
    """Compute the curl ∇×u (for vector fields, 2D/3D only)."""
    # Check input dimensions
    if self.dim == 1:
        raise ValueError("curl not defined for 1D")
    elif self.dim == 2 and u.shape[-3] != 2:
        raise ValueError("For 2D, input must have 2 components in the vector dimension")
    elif self.dim == 3 and u.shape[-4] != 3:
        raise ValueError("For 3D, input must have 3 components in the vector dimension")
    if self.dim == 2:
        # In 2D: ∂v/∂x - ∂u/∂y where u = (u,v) is a 2D vector field
        return self.dx(u[..., 1, :, :]) - self.dy(u[..., 0, :, :])
    elif self.dim == 3:
        # In 3D, ∇×u = (∂w/∂y - ∂v/∂z, ∂u/∂z - ∂w/∂x, ∂v/∂x - ∂u/∂y) where u = (u,v,w) is a 3D vector field
        curl_x = self.dy(u[..., 2, :, :, :]) - self.dz(u[..., 1, :, :, :])  # ∂w/∂y - ∂v/∂z
        curl_y = self.dz(u[..., 0, :, :, :]) - self.dx(u[..., 2, :, :, :])  # ∂u/∂z - ∂w/∂x  
        curl_z = self.dx(u[..., 1, :, :, :]) - self.dy(u[..., 0, :, :, :])  # ∂v/∂x - ∂u/∂y
        
        # Stack the components into a 3D vector field
        return jnp.stack([curl_x, curl_y, curl_z], axis=-4)



"""
differentiation.py implements utilities for computing derivatives via finite-differences
and Fourier/spectral differentiation methods in JAX.
"""

class FiniteDiff:
    """A unified class for computing finite differences in 1D, 2D, or 3D using JAX."""

    def __init__(
        self, dim, h=1.0, periodic_in_x=True, periodic_in_y=True, periodic_in_z=True
    ):
        if dim not in [1, 2, 3]:
            raise ValueError("dim must be 1, 2, or 3")

        self.dim = dim

        if isinstance(h, (int, float)):
            self.h = tuple(h for _ in range(dim))
        else:
            if len(h) != dim:
                raise ValueError(f"For {dim}D, h must be a float or a tuple of length {dim}")
            self.h = tuple(h)

        self.periodic_in_x = periodic_in_x
        if dim >= 2:
            self.periodic_in_y = periodic_in_y
        if dim >= 3:
            self.periodic_in_z = periodic_in_z

    def dx(self, u, order=1):
        if self.dim == 1:
            return self._dx_1d(u, order)
        elif self.dim == 2:
            return self._dx_2d(u, order)
        else:  
            return self._dx_3d(u, order)

    def dy(self, u, order=1):
        if self.dim < 2:
            raise ValueError("dy is only available for 2D and 3D")
        elif self.dim == 2:
            return self._dy_2d(u, order)
        else:  
            return self._dy_3d(u, order)

    def dz(self, u, order=1):
        if self.dim < 3:
            raise ValueError("dz is only available for 3D")
        return self._dz_3d(u, order)

    def laplacian(self, u):
        if self.dim == 1:
            return self._dx_1d(u, 2)
        elif self.dim == 2:
            return self._dx_2d(u, 2) + self._dy_2d(u, 2)
        else:  
            return self._dx_3d(u, 2) + self._dy_3d(u, 2) + self._dz_3d(u, 2)

    def gradient(self, u):
        if self.dim == 1:
            return self._dx_1d(u, 1)
        elif self.dim == 2:
            grad_x = self._dx_2d(u, 1)
            grad_y = self._dy_2d(u, 1)
            return jnp.stack([grad_x, grad_y], axis=-3)
        else:  
            grad_x = self._dx_3d(u, 1)
            grad_y = self._dy_3d(u, 1)
            grad_z = self._dz_3d(u, 1)
            return jnp.stack([grad_x, grad_y, grad_z], axis=-4)

    def divergence(self, u):
        n_components_expected = self.dim
        n_components_actual = u.shape[-self.dim - 1]
        if n_components_actual != n_components_expected:
            raise ValueError(f"Input must be a {self.dim}D vector field with {n_components_expected} components")

        if self.dim == 1:
            return self._dx_1d(u[..., 0, :], 1)
        elif self.dim == 2:
            u1, u2 = u[..., 0, :, :], u[..., 1, :, :]
            return self._dx_2d(u1, 1) + self._dy_2d(u2, 1)
        else:  
            u1, u2, u3 = u[..., 0, :, :, :], u[..., 1, :, :, :], u[..., 2, :, :, :]
            return self._dx_3d(u1, 1) + self._dy_3d(u2, 1) + self._dz_3d(u3, 1)

    def curl(self, u):
        if self.dim == 1:
            raise ValueError("Curl is not defined for 1D")
        elif self.dim == 2:
            if u.shape[-3] != 2:
                raise ValueError("Input must be a 2D vector field with 2 components")
            u1, u2 = u[..., 0, :, :], u[..., 1, :, :]
            return self._dx_2d(u2, 1) - self._dy_2d(u1, 1)
        else:  
            if u.shape[-4] != 3:
                raise ValueError("Input must be a 3D vector field with 3 components")
            u1, u2, u3 = u[..., 0, :, :, :], u[..., 1, :, :, :], u[..., 2, :, :, :]
            curl_x = self._dy_3d(u3, 1) - self._dz_3d(u2, 1)
            curl_y = self._dz_3d(u1, 1) - self._dx_3d(u3, 1)
            curl_z = self._dx_3d(u2, 1) - self._dy_3d(u1, 1)
            return jnp.stack([curl_x, curl_y, curl_z], axis=-4)

    # --- 1D Derivatives ---
    def _dx_1d(self, u, order):
        if order == 1:
            return self._dx_1st_1d(u)
        elif order == 2:
            return self._dx_2nd_1d(u)
        else:
            raise ValueError("Only 1st and 2nd order derivatives currently supported")

    def _dx_1st_1d(self, u):
        if self.periodic_in_x:
            dx = (jnp.roll(u, -1, axis=-1) - jnp.roll(u, 1, axis=-1)) / (2.0 * self.h[0])
        else:
            dx = jnp.zeros_like(u)
            dx = dx.at[..., 1:-1].set((u[..., 2:] - u[..., :-2]) / (2.0 * self.h[0]))
            dx = dx.at[..., 0].set((-11 * u[..., 0] + 18 * u[..., 1] - 9 * u[..., 2] + 2 * u[..., 3]) / (6.0 * self.h[0]))
            dx = dx.at[..., -1].set((-2 * u[..., -4] + 9 * u[..., -3] - 18 * u[..., -2] + 11 * u[..., -1]) / (6.0 * self.h[0]))
        return dx

    def _dx_2nd_1d(self, u):
        if self.periodic_in_x:
            dxx = (jnp.roll(u, -1, axis=-1) - 2 * u + jnp.roll(u, 1, axis=-1)) / (self.h[0]**2)
        else:
            dxx = jnp.zeros_like(u)
            dxx = dxx.at[..., 1:-1].set((u[..., 2:] - 2 * u[..., 1:-1] + u[..., :-2]) / (self.h[0]**2))
            dxx = dxx.at[..., 0].set((2 * u[..., 0] - 5 * u[..., 1] + 4 * u[..., 2] - u[..., 3]) / (self.h[0]**2))
            dxx = dxx.at[..., -1].set((-u[..., -4] + 4 * u[..., -3] - 5 * u[..., -2] + 2 * u[..., -1]) / (self.h[0]**2))
        return dxx

    # --- 2D Derivatives ---
    def _dx_2d(self, u, order):
        if order == 1: return self._dx_1st_2d(u)
        elif order == 2: return self._dx_2nd_2d(u)
        else: raise ValueError("Only 1st and 2nd order derivatives currently supported")

    def _dy_2d(self, u, order):
        if order == 1: return self._dy_1st_2d(u)
        elif order == 2: return self._dy_2nd_2d(u)
        else: raise ValueError("Only 1st and 2nd order derivatives currently supported")

    def _dx_1st_2d(self, u):
        if self.periodic_in_x:
            dx = (jnp.roll(u, -1, axis=-2) - jnp.roll(u, 1, axis=-2)) / (2.0 * self.h[0])
        else:
            dx = jnp.zeros_like(u)
            dx = dx.at[..., 1:-1, :].set((u[..., 2:, :] - u[..., :-2, :]) / (2.0 * self.h[0]))
            dx = dx.at[..., 0, :].set((-11 * u[..., 0, :] + 18 * u[..., 1, :] - 9 * u[..., 2, :] + 2 * u[..., 3, :]) / (6.0 * self.h[0]))
            dx = dx.at[..., -1, :].set((-2 * u[..., -4, :] + 9 * u[..., -3, :] - 18 * u[..., -2, :] + 11 * u[..., -1, :]) / (6.0 * self.h[0]))
        return dx

    def _dy_1st_2d(self, u):
        if self.periodic_in_y:
            dy = (jnp.roll(u, -1, axis=-1) - jnp.roll(u, 1, axis=-1)) / (2.0 * self.h[1])
        else:
            dy = jnp.zeros_like(u)
            dy = dy.at[..., :, 1:-1].set((u[..., :, 2:] - u[..., :, :-2]) / (2.0 * self.h[1]))
            dy = dy.at[..., :, 0].set((-11 * u[..., :, 0] + 18 * u[..., :, 1] - 9 * u[..., :, 2] + 2 * u[..., :, 3]) / (6.0 * self.h[1]))
            dy = dy.at[..., :, -1].set((-2 * u[..., :, -4] + 9 * u[..., :, -3] - 18 * u[..., :, -2] + 11 * u[..., :, -1]) / (6.0 * self.h[1]))
        return dy

    def _dx_2nd_2d(self, u):
        if self.periodic_in_x:
            dxx = (jnp.roll(u, -1, axis=-2) - 2 * u + jnp.roll(u, 1, axis=-2)) / (self.h[0] ** 2)
        else:
            dxx = jnp.zeros_like(u)
            dxx = dxx.at[..., 1:-1, :].set((u[..., 2:, :] - 2 * u[..., 1:-1, :] + u[..., :-2, :]) / (self.h[0] ** 2))
            dxx = dxx.at[..., 0, :].set((2 * u[..., 0, :] - 5 * u[..., 1, :] + 4 * u[..., 2, :] - u[..., 3, :]) / (self.h[0] ** 2))
            dxx = dxx.at[..., -1, :].set((-u[..., -4, :] + 4 * u[..., -3, :] - 5 * u[..., -2, :] + 2 * u[..., -1, :]) / (self.h[0] ** 2))
        return dxx

    def _dy_2nd_2d(self, u):
        if self.periodic_in_y:
            dyy = (jnp.roll(u, -1, axis=-1) - 2 * u + jnp.roll(u, 1, axis=-1)) / (self.h[1] ** 2)
        else:
            dyy = jnp.zeros_like(u)
            dyy = dyy.at[..., :, 1:-1].set((u[..., :, 2:] - 2 * u[..., :, 1:-1] + u[..., :, :-2]) / (self.h[1] ** 2))
            dyy = dyy.at[..., :, 0].set((2 * u[..., :, 0] - 5 * u[..., :, 1] + 4 * u[..., :, 2] - u[..., :, 3]) / (self.h[1] ** 2))
            dyy = dyy.at[..., :, -1].set((-u[..., :, -4] + 4 * u[..., :, -3] - 5 * u[..., :, -2] + 2 * u[..., :, -1]) / (self.h[1] ** 2))
        return dyy

    # --- 3D Derivatives ---
    def _dx_3d(self, u, order):
        if order == 1: return self._dx_1st_3d(u)
        elif order == 2: return self._dx_2nd_3d(u)
        else: raise ValueError("Only 1st and 2nd order derivatives currently supported")

    def _dy_3d(self, u, order):
        if order == 1: return self._dy_1st_3d(u)
        elif order == 2: return self._dy_2nd_3d(u)
        else: raise ValueError("Only 1st and 2nd order derivatives currently supported")

    def _dz_3d(self, u, order):
        if order == 1: return self._dz_1st_3d(u)
        elif order == 2: return self._dz_2nd_3d(u)
        else: raise ValueError("Only 1st and 2nd order derivatives currently supported")

    def _dx_1st_3d(self, u):
        if self.periodic_in_x:
            dx = (jnp.roll(u, -1, axis=-3) - jnp.roll(u, 1, axis=-3)) / (2.0 * self.h[0])
        else:
            dx = jnp.zeros_like(u)
            dx = dx.at[..., 1:-1, :, :].set((u[..., 2:, :, :] - u[..., :-2, :, :]) / (2.0 * self.h[0]))
            dx = dx.at[..., 0, :, :].set((-11 * u[..., 0, :, :] + 18 * u[..., 1, :, :] - 9 * u[..., 2, :, :] + 2 * u[..., 3, :, :]) / (6.0 * self.h[0]))
            dx = dx.at[..., -1, :, :].set((-2 * u[..., -4, :, :] + 9 * u[..., -3, :, :] - 18 * u[..., -2, :, :] + 11 * u[..., -1, :, :]) / (6.0 * self.h[0]))
        return dx

    def _dy_1st_3d(self, u):
        if self.periodic_in_y:
            dy = (jnp.roll(u, -1, axis=-2) - jnp.roll(u, 1, axis=-2)) / (2.0 * self.h[1])
        else:
            dy = jnp.zeros_like(u)
            dy = dy.at[..., :, 1:-1, :].set((u[..., :, 2:, :] - u[..., :, :-2, :]) / (2.0 * self.h[1]))
            dy = dy.at[..., :, 0, :].set((-11 * u[..., :, 0, :] + 18 * u[..., :, 1, :] - 9 * u[..., :, 2, :] + 2 * u[..., :, 3, :]) / (6.0 * self.h[1]))
            dy = dy.at[..., :, -1, :].set((-2 * u[..., :, -4, :] + 9 * u[..., :, -3, :] - 18 * u[..., :, -2, :] + 11 * u[..., :, -1, :]) / (6.0 * self.h[1]))
        return dy

    def _dz_1st_3d(self, u):
        if self.periodic_in_z:
            dz = (jnp.roll(u, -1, axis=-1) - jnp.roll(u, 1, axis=-1)) / (2.0 * self.h[2])
        else:
            dz = jnp.zeros_like(u)
            dz = dz.at[..., :, :, 1:-1].set((u[..., :, :, 2:] - u[..., :, :, :-2]) / (2.0 * self.h[2]))
            dz = dz.at[..., :, :, 0].set((-11 * u[..., :, :, 0] + 18 * u[..., :, :, 1] - 9 * u[..., :, :, 2] + 2 * u[..., :, :, 3]) / (6.0 * self.h[2]))
            dz = dz.at[..., :, :, -1].set((-2 * u[..., :, :, -4] + 9 * u[..., :, :, -3] - 18 * u[..., :, :, -2] + 11 * u[..., :, :, -1]) / (6.0 * self.h[2]))
        return dz

    def _dx_2nd_3d(self, u):
        if self.periodic_in_x:
            dxx = (jnp.roll(u, -1, axis=-3) - 2 * u + jnp.roll(u, 1, axis=-3)) / (self.h[0] ** 2)
        else:
            dxx = jnp.zeros_like(u)
            dxx = dxx.at[..., 1:-1, :, :].set((u[..., 2:, :, :] - 2 * u[..., 1:-1, :, :] + u[..., :-2, :, :]) / (self.h[0] ** 2))
            dxx = dxx.at[..., 0, :, :].set((2 * u[..., 0, :, :] - 5 * u[..., 1, :, :] + 4 * u[..., 2, :, :] - u[..., 3, :, :]) / (self.h[0] ** 2))
            dxx = dxx.at[..., -1, :, :].set((-u[..., -4, :, :] + 4 * u[..., -3, :, :] - 5 * u[..., -2, :, :] + 2 * u[..., -1, :, :]) / (self.h[0] ** 2))
        return dxx

    def _dy_2nd_3d(self, u):
        if self.periodic_in_y:
            dyy = (jnp.roll(u, -1, axis=-2) - 2 * u + jnp.roll(u, 1, axis=-2)) / (self.h[1] ** 2)
        else:
            dyy = jnp.zeros_like(u)
            dyy = dyy.at[..., :, 1:-1, :].set((u[..., :, 2:, :] - 2 * u[..., :, 1:-1, :] + u[..., :, :-2, :]) / (self.h[1] ** 2))
            dyy = dyy.at[..., :, 0, :].set((2 * u[..., :, 0, :] - 5 * u[..., :, 1, :] + 4 * u[..., :, 2, :] - u[..., :, 3, :]) / (self.h[1] ** 2))
            dyy = dyy.at[..., :, -1, :].set((-u[..., :, -4, :] + 4 * u[..., :, -3, :] - 5 * u[..., :, -2, :] + 2 * u[..., :, -1, :]) / (self.h[1] ** 2))
        return dyy

    def _dz_2nd_3d(self, u):
        if self.periodic_in_z:
            dzz = (jnp.roll(u, -1, axis=-1) - 2 * u + jnp.roll(u, 1, axis=-1)) / (self.h[2] ** 2)
        else:
            dzz = jnp.zeros_like(u)
            dzz = dzz.at[..., :, :, 1:-1].set((u[..., :, :, 2:] - 2 * u[..., :, :, 1:-1] + u[..., :, :, :-2]) / (self.h[2] ** 2))
            dzz = dzz.at[..., :, :, 0].set((2 * u[..., :, :, 0] - 5 * u[..., :, :, 1] + 4 * u[..., :, :, 2] - u[..., :, :, 3]) / (self.h[2] ** 2))
            dzz = dzz.at[..., :, :, -1].set((-u[..., :, :, -4] + 4 * u[..., :, :, -3] - 5 * u[..., :, :, -2] + 2 * u[..., :, :, -1]) / (self.h[2] ** 2))
        return dzz


# --- Backward compatibility functions ---
def central_diff_1d(x, h, periodic_in_x=True):
    warnings.warn(
        "central_diff_1d is deprecated. Please use FiniteDiff class instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    fd1d = FiniteDiff(dim=1, h=h, periodic_in_x=periodic_in_x)
    return fd1d.dx(x)

def central_diff_2d(x, h, periodic_in_x=True, periodic_in_y=True):
    warnings.warn(
        "central_diff_2d is deprecated. Please use FiniteDiff class instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    fd2d = FiniteDiff(dim=2, h=h, periodic_in_x=periodic_in_x, periodic_in_y=periodic_in_y)
    return fd2d.dx(x), fd2d.dy(x)

def central_diff_3d(x, h, periodic_in_x=True, periodic_in_y=True, periodic_in_z=True):
    warnings.warn(
        "central_diff_3d is deprecated. Please use FiniteDiff class instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    fd3d = FiniteDiff(dim=3, h=h, periodic_in_x=periodic_in_x, periodic_in_y=periodic_in_y, periodic_in_z=periodic_in_z)
    return fd3d.dx(x), fd3d.dy(x), fd3d.dz(x)


class FourierDiff:
    """A unified class for computing Fourier/spectral derivatives in 1D, 2D, 3D using JAX."""

    def __init__(
        self,
        dim,
        L=None,
        use_fc=False,
        fc_degree=4,
        fc_n_additional_pts=50,
        low_pass_filter_ratio=None,
    ):
        if dim not in [1, 2, 3]:
            raise ValueError("dim must be 1, 2, or 3")

        self.dim = dim

        if L is None:
            L = 2 * jnp.pi
        if not isinstance(L, (tuple, list)):
            L = (L,) * dim
        if len(L) != dim:
            raise ValueError(f"For {dim}D, L must be a single float or tuple with {dim} elements")
        self.L = L[0] if dim == 1 else L

        self.use_fc = use_fc
        self.fc_degree = fc_degree
        self.fc_n_additional_pts = fc_n_additional_pts
        self.low_pass_filter_ratio = low_pass_filter_ratio

        self.FC = None
        if self.use_fc:
            if self.use_fc.lower() in ['legendre', 'gram']:
                FC_class = FCLegendre if self.use_fc.lower() == 'legendre' else FCGram
                self.FC = FC_class(d=self.fc_degree, n_additional_pts=self.fc_n_additional_pts)
            else:
                raise ValueError(f"Given FC input {self.use_fc} is not valid. Must be 'legendre' or 'gram'.")

    def compute_multiple_derivatives(self, u, derivatives):
        if self.dim == 1:
            return self._compute_multiple_derivatives_1d(u, derivatives)
        elif self.dim == 2:
            return self._compute_multiple_derivatives_2d(u, derivatives)
        elif self.dim == 3:
            return self._compute_multiple_derivatives_3d(u, derivatives)

    def derivative(self, u, order):
        if len(order) != self.dim:
            raise ValueError(f"For {self.dim}D, order must be a tuple with {self.dim} elements")
            
        if self.dim == 1:
            derivatives = self._compute_multiple_derivatives_1d(u, [order[0]])
        elif self.dim == 2:
            derivatives = self._compute_multiple_derivatives_2d(u, [order])
        elif self.dim == 3:
            derivatives = self._compute_multiple_derivatives_3d(u, [order])

        return derivatives[0]

    def partial(self, u, direction="x", order=1):
        if direction == "x":
            return self.dx(u, order=order)
        elif direction == "y" and self.dim >= 2:
            return self.dy(u, order=order)
        elif direction == "z" and self.dim >= 3:
            return self.dz(u, order=order)
        else:
            raise ValueError(f"Invalid direction '{direction}' for dimension {self.dim}")

    def dx(self, u, order=1):
        if self.dim == 1: return self._dx_1d(u, order)
        elif self.dim == 2: return self._dx_2d(u, order)
        elif self.dim == 3: return self._dx_3d(u, order)

    def dy(self, u, order=1):
        if self.dim < 2: raise ValueError("dy method only available for 2D and 3D")
        elif self.dim == 2: return self._dy_2d(u, order)
        elif self.dim == 3: return self._dy_3d(u, order)

    def dz(self, u, order=1):
        if self.dim < 3: raise ValueError("dz method only available for 3D")
        return self._dz_3d(u, order)

    def laplacian(self, u):
        if self.dim == 1: return self.dx(u, order=2)
        elif self.dim == 2: return self.dx(u, order=2) + self.dy(u, order=2)
        elif self.dim == 3: return self.dx(u, order=2) + self.dy(u, order=2) + self.dz(u, order=2)

    def gradient(self, u):
        if self.dim == 1:
            return jnp.expand_dims(self.dx(u), axis=-2)
        elif self.dim == 2:
            return jnp.stack([self.dx(u), self.dy(u)], axis=-3)
        elif self.dim == 3:
            return jnp.stack([self.dx(u), self.dy(u), self.dz(u)], axis=-4)

    def divergence(self, u):
        expected_dims = {1: 1, 2: 2, 3: 3}
        if u.shape[-self.dim - 1] != expected_dims[self.dim]:
            raise ValueError(f"For {self.dim}D, input must have {expected_dims[self.dim]} components in the vector dimension")

        if self.dim == 1:
            return self.dx(jnp.squeeze(u, axis=-2))
        elif self.dim == 2:
            return self.dx(u[..., 0, :, :]) + self.dy(u[..., 1, :, :])
        elif self.dim == 3:
            return self.dx(u[..., 0, :, :, :]) + self.dy(u[..., 1, :, :, :]) + self.dz(u[..., 2, :, :, :])
    
    def curl(self, u):
        if self.dim == 1:
            raise ValueError("curl not defined for 1D")
        elif self.dim == 2 and u.shape[-3] != 2:
            raise ValueError("For 2D, input must have 2 components in the vector dimension")
        elif self.dim == 3 and u.shape[-4] != 3:
            raise ValueError("For 3D, input must have 3 components in the vector dimension")
        
        if self.dim == 2:
            return self.dx(u[..., 1, :, :]) - self.dy(u[..., 0, :, :])
        elif self.dim == 3:
            curl_x = self.dy(u[..., 2, :, :, :]) - self.dz(u[..., 1, :, :, :])
            curl_y = self.dz(u[..., 0, :, :, :]) - self.dx(u[..., 2, :, :, :])  
            curl_z = self.dx(u[..., 1, :, :, :]) - self.dy(u[..., 0, :, :, :])
            return jnp.stack([curl_x, curl_y, curl_z], axis=-4)

    def _compute_multiple_derivatives_1d(self, u, orders):
        if u is None:
            raise ValueError("Input tensor u is None")

        L_x = self.L
        nx = u.shape[-1]
        u_clone = u

        if self.use_fc and self.FC is not None:
            # Requires FC component to be JAX compatible as well
            u_clone = self.FC(u_clone, dim=1)
            L_x *= (nx + self.fc_n_additional_pts) / nx

        nx = u_clone.shape[-1]
        dx = L_x / nx

        u_h = jnp.fft.rfft(u_clone, axis=-1)
        k_x = jnp.fft.rfftfreq(nx, d=dx) * (2 * jnp.pi)

        if self.low_pass_filter_ratio is not None:
            cutoff = int(u_h.shape[-1] * self.low_pass_filter_ratio)
            u_h = u_h.at[..., cutoff:].set(0)

        results = []
        for order in orders:
            derivative_u_h = ((1j * k_x) ** order) * u_h
            results.append(derivative_u_h)

        derivatives_ft = jnp.stack(results, axis=0)
        derivatives_real = jnp.fft.irfft(derivatives_ft, axis=-1, n=nx)

        if self.use_fc and self.FC is not None:
            derivatives_real = self.FC.restrict(derivatives_real, dim=1)

        return [derivatives_real[i] for i in range(len(orders))]

    def _dx_1d(self, u, order):
        derivatives = self._compute_multiple_derivatives_1d(u, [order])
        return derivatives[0]

    def _compute_multiple_derivatives_2d(self, u, derivatives):
        if u is None:
            raise ValueError("Input tensor u is None")

        L_x, L_y = self.L[0], self.L[1]
        nx, ny = u.shape[-2], u.shape[-1]
        u_clone = u

        if self.use_fc and self.FC is not None:
            u_clone = self.FC(u_clone, dim=2)
            L_x *= (nx + self.fc_n_additional_pts) / nx
            L_y *= (ny + self.fc_n_additional_pts) / ny

        nx, ny = u_clone.shape[-2], u_clone.shape[-1]
        dx, dy = L_x / nx, L_y / ny

        u_h = jnp.fft.fft2(u_clone, axes=(-2, -1))

        k_x = jnp.fft.fftfreq(nx, d=dx) * (2 * jnp.pi)
        k_y = jnp.fft.fftfreq(ny, d=dy) * (2 * jnp.pi)

        KX, KY = jnp.meshgrid(k_x, k_y, indexing="ij")

        if self.low_pass_filter_ratio is not None:
            cutoff_x = int(nx * self.low_pass_filter_ratio)
            cutoff_y = int(ny * self.low_pass_filter_ratio)
            u_h = u_h.at[..., cutoff_y:, :].set(0)
            u_h = u_h.at[..., :, cutoff_x:].set(0)

        results = []
        for order_x, order_y in derivatives:
            KX_expanded = jnp.broadcast_to(KX, u_h.shape)
            KY_expanded = jnp.broadcast_to(KY, u_h.shape)
            
            derivative_u_h = ((1j * KX_expanded) ** order_x) * ((1j * KY_expanded) ** order_y) * u_h
            results.append(derivative_u_h)

        derivatives_ft = jnp.stack(results, axis=0)
        derivatives_real = jnp.fft.ifft2(derivatives_ft, axes=(-2, -1)).real

        if self.use_fc and self.FC is not None:
            derivatives_real = self.FC.restrict(derivatives_real, dim=2)

        return [derivatives_real[i] for i in range(len(derivatives))]

    def _dx_2d(self, u, order):
        derivatives = self._compute_multiple_derivatives_2d(u, [(order, 0)])
        return derivatives[0]

    def _dy_2d(self, u, order):
        derivatives = self._compute_multiple_derivatives_2d(u, [(0, order)])
        return derivatives[0]

    def _compute_multiple_derivatives_3d(self, u, derivatives):
        if u is None:
            raise ValueError("Input tensor u is None")

        L_x, L_y, L_z = self.L[0], self.L[1], self.L[2]
        nx, ny, nz = u.shape[-3], u.shape[-2], u.shape[-1]
        u_clone = u

        if self.use_fc and self.FC is not None:
            u_clone = self.FC(u_clone, dim=3)
            L_x *= (nx + self.fc_n_additional_pts) / nx
            L_y *= (ny + self.fc_n_additional_pts) / ny
            L_z *= (nz + self.fc_n_additional_pts) / nz

        nx, ny, nz = u_clone.shape[-3], u_clone.shape[-2], u_clone.shape[-1]
        dx, dy, dz = L_x / nx, L_y / ny, L_z / nz

        u_h = jnp.fft.fftn(u_clone, axes=(-3, -2, -1))

        k_x = jnp.fft.fftfreq(nx, d=dx) * (2 * jnp.pi)
        k_y = jnp.fft.fftfreq(ny, d=dy) * (2 * jnp.pi)
        k_z = jnp.fft.fftfreq(nz, d=dz) * (2 * jnp.pi)

        KX, KY, KZ = jnp.meshgrid(k_x, k_y, k_z, indexing="ij")

        if self.low_pass_filter_ratio is not None:
            cutoff_x = int(nx * self.low_pass_filter_ratio)
            cutoff_y = int(ny * self.low_pass_filter_ratio)
            cutoff_z = int(nz * self.low_pass_filter_ratio)
            u_h = u_h.at[..., cutoff_y:, :, :].set(0)
            u_h = u_h.at[..., :, cutoff_x:, :].set(0)
            u_h = u_h.at[..., :, :, cutoff_z:].set(0)

        results = []
        for order_x, order_y, order_z in derivatives:
            KX_expanded = jnp.broadcast_to(KX, u_h.shape)
            KY_expanded = jnp.broadcast_to(KY, u_h.shape)
            KZ_expanded = jnp.broadcast_to(KZ, u_h.shape)
            
            derivative_u_h = ((1j * KX_expanded) ** order_x) * ((1j * KY_expanded) ** order_y) * ((1j * KZ_expanded) ** order_z) * u_h
            results.append(derivative_u_h)

        derivatives_ft = jnp.stack(results, axis=0)
        derivatives_real = jnp.fft.ifftn(derivatives_ft, axes=(-3, -2, -1)).real

        if self.use_fc and self.FC is not None:
            derivatives_real = self.FC.restrict(derivatives_real, dim=3)

        return [derivatives_real[i] for i in range(len(derivatives))]

    def _dx_3d(self, u, order):
        derivatives = self._compute_multiple_derivatives_3d(u, [(order, 0, 0)])
        return derivatives[0]

    def _dy_3d(self, u, order):
        derivatives = self._compute_multiple_derivatives_3d(u, [(0, order, 0)])
        return derivatives[0]

    def _dz_3d(self, u, order):
        derivatives = self._compute_multiple_derivatives_3d(u, [(0, 0, order)])
        return derivatives[0]