import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional

from .channel_mlp_jax import LinearChannelMLP
from .segment_csr_jax import segment_csr


class IntegralTransform(nn.Module):
    """Integral Kernel Transform (GNO).

    It computes one of the following:
        (a) \\int_{A(x)} k(x, y) dy
        (b) \\int_{A(x)} k(x, y) * f(y) dy
        (c) \\int_{A(x)} k(x, y, f(y)) dy
        (d) \\int_{A(x)} k(x, y, f(y)) * f(y) dy

    x : Points for which the output is defined

    y : Points for which the input is defined
    A(x) : A subset of all points y (depending on\
        each x) over which to integrate

    k : A kernel parametrized as a MLP (LinearChannelMLP)

    f : Input function to integrate against given\
        on the points y

    If f is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(x) are specified
    as a graph in CRS format.

    Parameters
    ----------
    channel_mlp : nn.Module, optional
        MLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f.
        MLP should not be pointwise and should only operate across
        channels to preserve the discretization-invariance of the
        kernel integral, by default None
    channel_mlp_layers : list, optional
        List of layers sizes speficing a MLP which
        parametrizes the kernel k. The MLP will be
        instansiated by the LinearChannelMLP class, by default None
    channel_mlp_non_linearity : callable, optional
        Non-linear function used to be used by the
        LinearChannelMLP class, by default nn.gelu. Only used if channel_mlp_layers is
        given and channel_mlp is None, by default nn.gelu
    transform_type : str, optional
        Which integral transform to compute. Options: 'linear_kernelonly', 'linear', 'nonlinear_kernelonly', 'nonlinear'.
        The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b)
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter, by default 'linear'
    use_torch_scatter : bool, optional
        Whether to use torch-scatter to perform grouped reductions in the IntegralTransform.
        If False, uses native Python reduction in scirex.operators.layers.segment_csr, by default True

        .. warning::

            torch-scatter is an optional dependency that conflicts with the newest versions of PyTorch,
            so you must handle the conflict explicitly in your environment. See :ref:`torch_scatter_dependency`
            for more information.
    """

    channel_mlp: Any = None
    channel_mlp_layers: Any = None
    channel_mlp_non_linearity: Callable = nn.gelu
    transform_type: str = "linear"
    weighting_fn: Any = None
    reduction: str = "sum"
    use_torch_scatter: bool = True

    def setup(self):
        assert self.channel_mlp is not None or self.channel_mlp_layers is not None

        if self.transform_type not in (
            "linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"
        ):
            raise ValueError(
                f"Got transform_type={self.transform_type} but expected one of "
                "[linear_kernelonly, linear, nonlinear_kernelonly, nonlinear]"
            )

        if self.channel_mlp is None:
            self._channel_mlp = LinearChannelMLP(
                layers=self.channel_mlp_layers,
                non_linearity=self.channel_mlp_non_linearity,
            )
        else:
            self._channel_mlp = self.channel_mlp

    def __call__(self, y, neighbors, x=None, f_y=None, weights=None):
        """Compute a kernel integral transform. Assumes x=y if not specified.

        Integral is taken w.r.t. the neighbors.

        If no weights are given, a Monte-Carlo approximation is made.

        .. note :: For transforms of type 0 or 2, out channels must be
            the same as the channels of f

        Parameters
        ----------
        y : jnp.ndarray of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
            If batched, these must remain constant
            over the whole batch so no batch dim is needed.
        neighbors : dict
            The sets :math:`A(x)` given in CRS format. The
            dict must contain the keys "neighbors_index"
            and "neighbors_row_splits." For descriptions
            of the two, see NeighborSearch.
            If batch > 1, the neighbors must be constant
            across the entire batch.
        x : jnp.ndarray of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined. If None,
            x = y.
        f_y : jnp.ndarray of shape [batch, n, d3] or [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : jnp.ndarray of shape [n,], default None
            Weights for each point y proprtional to the
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are y_{j+1} - y_j. If None,
            :math:`1/|A(x)|` is used.

        Returns
        -------
        out_features : jnp.ndarray of shape [batch, m, d4] or [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """

        print(f"\n[IntegralTransform.__call__] ---- START ----")
        print(f"  transform_type  = {self.transform_type}")
        print(f"  reduction       = {self.reduction}")
        print(f"  weighting_fn    = {self.weighting_fn}")
        print(f"  y.shape         = {tuple(y.shape)}   (source/grid points)")
        print(f"  x.shape         = {tuple(x.shape) if x is not None else 'None (will be set to y)'}")
        print(f"  f_y.shape       = {tuple(f_y.shape) if f_y is not None else 'None'}")
        print(f"  weights         = {'provided, shape=' + str(tuple(weights.shape)) if weights is not None else 'None'}")
        print(f"  neighbors_index.shape      = {tuple(neighbors['neighbors_index'].shape)}")
        print(f"  neighbors_row_splits.shape = {tuple(neighbors['neighbors_row_splits'].shape)}")
        print(f"  total edges (neighbors_index length) = {neighbors['neighbors_index'].shape[0]}")

        if x is None:
            x = y
            print(f"  [branch] x was None -> x set to y, shape={tuple(x.shape)}")
        else:
            print(f"  [branch] x provided, shape={tuple(x.shape)}")

        rep_features = y[neighbors["neighbors_index"]]
        print(f"\n  rep_features = y[neighbors_index]")
        print(f"  rep_features.shape = {tuple(rep_features.shape)}   [total_edges, D_y]")
        print(f"  rep_features[:3]   =\n{rep_features[:3]}")

        # batching only matters if f_y (latent embedding) values are provided
        batched = False
        # f_y has a batch dim IFF batched=True
        if f_y is not None:
            if f_y.ndim == 3:
                batched = True
                batch_size = f_y.shape[0]
                in_features = f_y[:, neighbors["neighbors_index"], :]
                print(f"\n  [branch] f_y.ndim=3 -> batched=True, batch_size={batch_size}")
                print(f"  in_features = f_y[:, neighbors_index, :]")
                print(f"  in_features.shape = {tuple(in_features.shape)}   [batch, total_edges, D_f]")
            elif f_y.ndim == 2:
                batched = False
                in_features = f_y[neighbors["neighbors_index"]]
                print(f"\n  [branch] f_y.ndim=2 -> batched=False")
                print(f"  in_features.shape = {tuple(in_features.shape)}   [total_edges, D_f]")
        else:
            print(f"\n  [branch] f_y is None -> in_features not computed")

        num_reps = (
            neighbors["neighbors_row_splits"][1:]
            - neighbors["neighbors_row_splits"][:-1]
        )
        print(f"\n  num_reps (neighbors per query point):")
        print(f"  num_reps.shape = {tuple(num_reps.shape)}   [N_x]")
        print(f"  min={num_reps.min().item()}  max={num_reps.max().item()}  "
              f"mean={float(num_reps.mean()):.2f}  sum={num_reps.sum().item()}")

        self_features = jnp.repeat(x, num_reps, axis=0)
        print(f"\n  self_features = repeat(x, num_reps, axis=0)")
        print(f"  self_features.shape = {tuple(self_features.shape)}   [total_edges, D_x]")
        print(f"  self_features[:3]   =\n{self_features[:3]}")

        agg_features = jnp.concatenate([rep_features, self_features], axis=-1)
        print(f"\n  agg_features = concatenate([rep_features, self_features], axis=-1)")
        print(f"  agg_features.shape = {tuple(agg_features.shape)}   [total_edges, D_y+D_x]")

        if f_y is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            print(f"\n  [branch] transform_type='{self.transform_type}' -> concatenating f_y onto agg_features")
            if batched:
                agg_features = jnp.tile(agg_features, [batch_size] + [1] * agg_features.ndim)
                print(f"  [branch] batched -> agg_features tiled, shape={tuple(agg_features.shape)}")
            agg_features = jnp.concatenate([agg_features, in_features], axis=-1)
            print(f"  agg_features (after cat f_y).shape = {tuple(agg_features.shape)}   [total_edges, D_y+D_x+D_f]")
        else:
            print(f"\n  [branch] transform_type='{self.transform_type}' -> f_y NOT concatenated onto agg_features")

        rep_features = self._channel_mlp(agg_features)
        print(f"\n  rep_features = channel_mlp(agg_features)")
        print(f"  rep_features.shape = {tuple(rep_features.shape)}   [total_edges, D_out]  (kernel output per edge)")
        print(f"  rep_features[:3]   =\n{rep_features[:3]}")

        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            print(f"\n  [branch] transform_type='{self.transform_type}' and f_y provided -> multiply rep_features * in_features")
            if rep_features.ndim == 2 and batched:
                rep_features = jnp.tile(jnp.expand_dims(rep_features, 0), [batch_size] + [1] * rep_features.ndim)
                print(f"  [branch] rep_features was 2d but batched -> expand+tile, shape={tuple(rep_features.shape)}")
            rep_features = rep_features * in_features
            print(f"  rep_features (after *= in_features).shape = {tuple(rep_features.shape)}   k(x,y)*f(y) per edge")
        else:
            print(f"\n  [branch] skipping mul_(in_features): "
                  f"f_y={'None' if f_y is None else 'provided'}, transform_type='{self.transform_type}'")

        # Weight neighbors in each neighborhood, first according to the neighbor search (mollified GNO)
        # and second according to individually-provided weights.
        nbr_weights = neighbors.get("weights")
        print(f"\n  neighbors.get('weights') = {'provided, shape=' + str(tuple(nbr_weights.shape)) if nbr_weights is not None else 'None'}")
        if nbr_weights is None:
            nbr_weights = weights
            print(f"  fallback to 'weights' arg = {'provided, shape=' + str(tuple(nbr_weights.shape)) if nbr_weights is not None else 'None'}")
        if nbr_weights is None and self.weighting_fn is not None:
            raise KeyError("if a weighting function is provided, your neighborhoods must contain weights.")
        if nbr_weights is not None:
            print(f"  [branch] nbr_weights provided -> applying weights")
            nbr_weights = jnp.expand_dims(jnp.expand_dims(nbr_weights, -1), 0)
            print(f"  nbr_weights (expand_dims).shape = {tuple(nbr_weights.shape)}")
            if self.weighting_fn is not None:
                nbr_weights = self.weighting_fn(nbr_weights)
                print(f"  [branch] weighting_fn applied, nbr_weights.shape = {tuple(nbr_weights.shape)}")
            else:
                print(f"  [branch] no weighting_fn, using raw weights")
            rep_features = rep_features * nbr_weights
            reduction = "sum"  # Force sum reduction for weighted GNO layers
            print(f"  rep_features (after *= nbr_weights).shape = {tuple(rep_features.shape)}")
            print(f"  reduction forced to 'sum'")
        else:
            reduction = self.reduction
            print(f"  [branch] no nbr_weights -> reduction = '{reduction}'")

        splits = neighbors["neighbors_row_splits"]
        print(f"\n  splits (row_splits) before batch expand: shape={tuple(splits.shape)}")
        if batched:
            splits = jnp.tile(jnp.expand_dims(splits, 0), [batch_size] + [1] * splits.ndim)
            print(f"  [branch] batched -> splits expanded, shape={tuple(splits.shape)}")
        else:
            print(f"  [branch] not batched -> splits unchanged")

        print(f"\n  calling segment_csr(rep_features, splits, reduction='{reduction}', use_scatter={self.use_torch_scatter})")
        print(f"  rep_features.shape = {tuple(rep_features.shape)}")
        print(f"  splits.shape       = {tuple(splits.shape)}")

        out_features = segment_csr(
            rep_features,
            splits,
            reduction=reduction,
            use_scatter=self.use_torch_scatter,
        )
        print(f"\n  out_features.shape = {tuple(out_features.shape)}   [batch, N_x, D_out] or [N_x, D_out]")
        print(f"  out_features[:, :3] =\n{out_features[:, :3] if out_features.ndim == 3 else out_features[:3]}")
        print(f"[IntegralTransform.__call__] ---- END ----\n")
        return out_features
