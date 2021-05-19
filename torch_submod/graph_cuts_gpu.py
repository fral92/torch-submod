"""Argmin-differentiable total variation functions."""
from __future__ import division, print_function

from pathos import multiprocessing
# Must import these first, gomp issues with pytorch.
from torch_proxtv import tv1w_2d

import numpy as np
from sklearn.isotonic import isotonic_regression as isotonic

import torch
from torch.autograd import Function
from .blocks import blockwise_means, blocks_2d

__all__ = ("TotalVariation2d", "TotalVariation2dWeighted", "TotalVariation1d")


class TotalVariationBase(Function):

    @staticmethod
    def _grad_x(opt, grad_output, average_connected):
        if opt.ndim == 1:
            opt = opt.reshape(1, -1)

        if average_connected:
            blocks = blocks_2d(opt)
        else:
            _, blocks = np.unique(opt.ravel(), return_inverse=True)
        grad_x = blockwise_means(blocks.ravel(), grad_output.ravel())
        # We need the clone as there seems to e a double-free error in py27,
        # namely, torch free()s the array after numpy has already free()d it.
        return grad_x.reshape(opt.shape)

    @staticmethod
    def _grad_w_row(opt, grad_x):
        """Compute the derivative with respect to the row weights."""
        diffs_row = torch.sign(opt[..., :-1] - opt[..., 1:])
        return - diffs_row * (grad_x[..., :-1] - grad_x[..., 1:])

    @staticmethod
    def _grad_w_col(opt, grad_x):
        """Compute the derivative with respect to the column weights."""
        diffs_col = torch.sign(opt[..., :-1, :] - opt[..., 1:, :])
        return - diffs_col * (grad_x[..., :-1, :] - grad_x[..., 1:, :])

    @staticmethod
    def _refine(opt, x, weights_row, weights_col):
        """Refine the solution by solving an isotonic regression.

        The weights can either be two-dimensional tensors, or of shape (1,)."""
        idx = np.argsort(opt.ravel())  # Will pick an arbitrary order cone.
        ordered_vec = np.zeros_like(idx, dtype=np.float)
        ordered_vec[idx] = np.arange(np.size(opt))
        f = TotalVariationBase._linearize(ordered_vec.reshape(opt.shape),
                                          weights_row,
                                          weights_col)
        opt_idx = isotonic((x.ravel() - f.ravel())[idx])
        opt = np.zeros_like(opt_idx)
        opt[idx] = opt_idx
        return opt

    @staticmethod
    def _linearize(y, weights_row, weights_col):
        """Compute a linearization of the graph-cut function at the given point.

        Arguments
        ---------
        y : numpy.ndarray
            The point where the linearization is computed, shape ``(m, n)``.
        weights_row : numpy.ndarray
            The non-negative row weights, with shape ``(m, n - 1)``.
        y : numpy.ndarray
            The non-negative column weights, with shape ``(m - 1, n)``.

        Returns
        -------
        numpy.ndarray
            The linearization of the graph-cut function at ``y``."""
        diffs_col = np.sign(y[1:, :] - y[:-1, :])
        diffs_row = np.sign(y[:, 1:] - y[:, :-1])

        f = np.zeros_like(y)  # The linearization.
        f[:, 1:] += diffs_row * weights_row
        f[:, :-1] -= diffs_row * weights_row
        f[1:, :] += diffs_col * weights_col
        f[:-1, :] -= diffs_col * weights_col

        return f


def TotalVariation2dWeighted(refine=True, average_connected=True,
                             num_workers=8, multiprocess=False, tv_args={}):
    r"""A two dimensional total variation function.

    Specifically, given as input the unaries `x`, positive row weights
    :math:`\mathbf{r}` and positive column weights :math:`\mathbf{c}`, the
    output is computed as

    .. math::

        \textrm{argmin}_{\mathbf z}
            \frac{1}{2} \|\mathbf{x}-\mathbf{z}\|^2 +
            \sum_{i, j} r_{i,j} |z_{i, j} - z_{i, j + 1}| +
            \sum_{i, j} c_{i,j} |z_{i, j} - z_{i + 1, j}|.

    Arguments
    ---------
        refine: bool
            If set the solution will be refined with isotonic regression.
        average_connected: bool
            How to compute the approximate derivative.

            If ``True``, will average within each connected component.
            If ``False``, it will average within each block of equal values.
            Typically, you want this set to true.
        tv_args: dict
            The dictionary of arguments passed to the total variation solver.
        """

    class TotalVariation2dWeighted_(TotalVariationBase):

        @staticmethod
        def solve_and_refine(x, w_col, w_row, refine=True, **tv_args):

            opt = tv1w_2d(x, w_col, w_row, **tv_args)
            if refine:
                opt = TotalVariationBase._refine(opt, x, w_row, w_col)

            return opt

        @staticmethod
        def _grad_x(opt, grad_output, average_connected):
            return TotalVariationBase._grad_x(opt, grad_output,
                                              average_connected)

        @staticmethod
        def forward(ctx, x, weights_row, weights_col):
            r"""Solve the total variation problem and return the solution.

            Arguments
            ---------
                x: :class:`torch:torch.Tensor`
                    A tensor with shape ``(m, n)`` holding the input signal.
                weights_row: :class:`torch:torch.Tensor`
                    The horizontal edge weights.

                    Tensor of shape ``(m, n - 1)``, or ``(1,)`` if all weights
                    are equal.
                weights_col: :class:`torch:torch.Tensor`
                    The vertical edge weights.

                    Tensor of shape ``(m - 1, n)``, or ``(1,)`` if all weights
                    are equal.

            Returns
            -------
            :class:`torch:torch.Tensor`
                The solution to the total variation problem, of shape ``(m, n)``.
            """
            opt = TotalVariation2dWeighted_.solve_and_refine(
                x, weights_col, weights_row,
                refine=refine, **tv_args).view_as(x)

            ctx.save_for_backward(opt)
            ctx.device = x.device
            return opt

        @staticmethod
        def backward(ctx, grad_output):
            opt, = ctx.saved_tensors
            grad_weights_row, grad_weights_col = None, None
            grad_x = TotalVariation2dWeighted_._grad_x(
                opt, grad_output, average_connected=average_connected)

            if ctx.needs_input_grad[1]:
                grad_weights_row = TotalVariation2dWeighted_._grad_w_row(
                    opt, grad_x)

            if ctx.needs_input_grad[2]:
                grad_weights_col = TotalVariation2dWeighted_._grad_w_col(
                    opt, grad_x)

            return grad_x, grad_weights_row, grad_weights_col

    return TotalVariation2dWeighted_.apply
