# -*- coding: utf-8 -*-
"""Visualization helpers for diffusion trajectories and image grids.

Originally from diffusion_utilities.py.
"""

from numpy.typing import ArrayLike

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Colormap

import torch
from torchvision.utils import make_grid
from torchvision.transforms import functional


def plot_image_grid(
    images: torch.Tensor,
    figsize: tuple,
    n_rows: int,
    n_cols: int,
    padding: int = 2,
    pad_value: float = 1.0,
    cmap: Colormap = "gray",
    normalize: bool = False,
    axis_on_off: str = "off",
):
    """Display a batch of images as a grid.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images ``(N, C, H, W)``.
    figsize : tuple
        Figure size passed to ``plt.subplots``.
    n_rows : int
        Number of rows (informational; grid layout determined by ``n_cols``).
    n_cols : int
        Number of columns in the grid.
    padding : int
        Pixels of padding between images.
    pad_value : float
        Fill value for padding.
    cmap : Colormap
        Colormap used by ``imshow``.
    normalize : bool
        Normalise pixel values to ``[0, 1]``.
    axis_on_off : str
        ``"on"`` or ``"off"``.

    Returns
    -------
    fig, ax
    """
    grid = make_grid(
        images,
        nrow=n_cols,
        padding=padding,
        normalize=normalize,
        pad_value=pad_value,
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(functional.to_pil_image(grid), cmap=cmap)
    ax.axis(axis_on_off)
    return fig, ax


def plot_image_evolution(
    images: torch.Tensor,
    n_images: int,
    n_intermediate_steps: ArrayLike,
    figsize: tuple,
    cmap: Colormap = "gray",
):
    """Plot a selection of time steps from a diffusion trajectory (grayscale).

    Parameters
    ----------
    images : torch.Tensor
        Shape ``(n_images, C, H, W, n_steps)``.
    n_images : int
        Number of rows (one per image sample).
    n_intermediate_steps : array-like
        Indices of the time steps to display.
    figsize : tuple
        Figure size.
    cmap : Colormap
        Colormap for ``imshow``.

    Returns
    -------
    fig, axs
    """
    fig, axs = plt.subplots(
        n_images,
        len(n_intermediate_steps),
        figsize=figsize,
    )
    for n_image in np.arange(n_images):
        for i, ax in enumerate(axs[n_image, :]):
            ax.imshow(
                images[n_image, 0, :, :, n_intermediate_steps[i]],
                cmap=cmap,
            )
            axs[n_image, i].set_axis_off()
    return fig, axs


def plot_image_evolution_rgb(
    images: torch.Tensor,
    n_images: int,
    n_intermediate_steps: ArrayLike,
    figsize: tuple,
):
    """Plot a selection of time steps from a diffusion trajectory (RGB).

    Parameters
    ----------
    images : torch.Tensor
        Shape ``(n_images, C, H, W, n_steps)``.
    n_images : int
        Number of rows.
    n_intermediate_steps : array-like
        Indices of the time steps to display.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, axs
    """
    fig, axs = plt.subplots(
        n_images,
        len(n_intermediate_steps),
        figsize=figsize,
    )
    for n_image in np.arange(n_images):
        for i, ax in enumerate(axs[n_image, :]):
            img = images[n_image, :, :, :, n_intermediate_steps[i]]  # (C, H, W)
            img = img - img.min()
            img = img / (img.max() + 1e-8)
            img = img.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_axis_off()
    return fig, axs


def animation_images(images_t, interval, figsize):
    """Create a matplotlib animation from a sequence of grayscale images.

    Parameters
    ----------
    images_t : array-like
        Shape ``(H, W, n_frames)``.
    interval : int
        Delay between frames in milliseconds.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, ax, anim : FuncAnimation
    """
    _, _, n_frames = np.shape(images_t)
    fig, ax = plt.subplots(figsize=figsize)
    img_display = ax.imshow(images_t[:, :, 0], cmap="gray")

    def update(t):
        img_display.set_array(images_t[:, :, t])
        return [img_display]

    return (
        fig,
        ax,
        animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval,
            blit=False,
        ),
    )
