import jax.numpy as jnp
import flax.linen as nn

# only import open3d if built
open3d_built = False
try:
    from open3d.ml.torch.layers import FixedRadiusSearch

    open3d_built = True
except:
    pass


# Uses open3d by default which, as of October 2024, requires torch 2.0 and cuda11.*
class NeighborSearch(nn.Module):
    """Neighborhood search between two arbitrary coordinate meshes.

    For each point `x` in `queries`, returns a set of the indices of all points `y` in `data`
    within the ball of radius r `B_r(x)`

    Parameters
    ----------
    use_open3d : bool, optional
        Whether to use open3d or native JAX implementation, by default True
        NOTE: open3d implementation requires 3d data
    return_norm : bool, optional
        Whether to return normalized distances, by default False
    """

    use_open3d: bool = True
    return_norm: bool = False

    def setup(self):
        if self.use_open3d and open3d_built:  # slightly faster, works on GPU in 3d only
            self.search_fn = FixedRadiusSearch()
            self._use_open3d = True
        else:  # slower fallback, works on GPU and CPU
            self.search_fn = None
            self._use_open3d = False

    def __call__(self, data, queries, radius):
        """
        Find the neighbors, in data, of each point in queries
        within a ball of radius. Returns in CRS format.

        Parameters
        ----------
        data : jnp.ndarray of shape [n, d]
            Search space of possible neighbors
            NOTE: open3d requires d=3
        queries : jnp.ndarray of shape [m, d]
            Points for which to find neighbors
            NOTE: open3d requires d=3
        radius : float
            Radius of each ball: B(queries[j], radius)

        Returns
        -------
        return_dict : dict
            Dictionary with keys: neighbors_index, neighbors_row_splits
                neighbors_index: jnp.ndarray with dtype=int64
                    Index of each neighbor in data for every point
                    in queries. Neighbors are ordered in the same orderings
                    as the points in queries. Open3d and torch_cluster
                    implementations can differ by a permutation of the
                    neighbors for every point.
                neighbors_row_splits: jnp.ndarray of shape [m+1] with dtype=int64
                    The value at index j is the sum of the number of
                    neighbors up to query point j-1. First element is 0
                    and last element is the total number of neighbors.
        """
        return_dict = {}

        if self._use_open3d:
            search_return = self.search_fn(data, queries, radius)
            return_dict["neighbors_index"] = jnp.array(search_return.neighbors_index, dtype=jnp.int64)
            return_dict["neighbors_row_splits"] = jnp.array(search_return.neighbors_row_splits, dtype=jnp.int64)
        else:
            return_dict = native_neighbor_search(data, queries, radius, self.return_norm)

        return return_dict


def native_neighbor_search(
    data: jnp.ndarray, queries: jnp.ndarray, radius: float, return_norm: bool = False
):
    """
    Native JAX implementation of a neighborhood search
    between two arbitrary coordinate meshes.

    Parameters
    ----------
    data : jnp.ndarray
        vector of data points from which to find neighbors
    queries : jnp.ndarray
        centers of neighborhoods
    radius : float
        size of each neighborhood
    """
    nbr_dict = {}

    # compute pairwise distances
    all_dists = jnp.linalg.norm(
        queries[:, None, :] - data[None, :, :], axis=-1
    )  # shaped num query points x num data points
    # keep zero-distance points
    eps = 1e-7
    all_dists = jnp.where(all_dists == 0.0, eps, all_dists)
    dists = jnp.where(all_dists <= radius, all_dists, 0.0)  # i,j is nonzero if j is i's neighbor
    nbr_indices = jnp.nonzero(dists, size=dists.size)[1]  # only keep the column indices
    # filter to only actual nonzero entries
    mask = dists.reshape(-1) > 0
    nbr_indices = nbr_indices[mask]

    if return_norm:
        weights = dists[dists > 0]
        nbr_dict["weights"] = weights ** 2  # weighting function computed on squared norms

    in_nbr = jnp.where(dists > 0, 1.0, 0.0)
    nbrhd_sizes = jnp.cumsum(jnp.sum(in_nbr, axis=1), axis=0)  # cumulative neighborhood sizes
    splits = jnp.concatenate((jnp.array([0.0]), nbrhd_sizes))

    nbr_dict["neighbors_index"] = nbr_indices.astype(jnp.int64)
    nbr_dict["neighbors_row_splits"] = splits.astype(jnp.int64)

    return nbr_dict
