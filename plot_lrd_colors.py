"""A script for plotting the color distribution of LRDs in FLARES."""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.cosmology import Planck15 as cosmo
from utils import get_synth_data, savefig

# Define the parser
parser = argparse.ArgumentParser(
    description="Plot the compactness of galaxies."
)
parser.add_argument(
    "--type",
    type=str,
    default="stellar",
    help="The type of data to plot.",
)

# Parse the arguments
args = parser.parse_args()

# Define the data file
if args.type == "stellar":
    data_file = "data/pure_stellar_<region>_<snap>.hdf5"
elif args.type == "agn":
    data_file = "data/pure_agn_<region>_<snap>.hdf5"
else:
    data_file = "data/combined_<region>_<snap>.hdf5"

# Get the synthesizer data
(
    _fluxes,
    _colors,
    _red1,
    _red2,
    _sizes,
    _masks,
    _indices,
    _weights,
) = get_synth_data(data_file, "attenuated", get_weights=True)

# Combine together the different snapshots (we'll plot everything together)
colors = {}
weights = []
masks = []
for snap in _colors.keys():
    for key in _colors[snap].keys():
        colors.setdefault(key, []).extend(_colors[snap])
    weights.extend(_weights[snap])
    masks.extend(_masks[snap])

# Get the lrd sample (where mask is true)
lrd_colors = {key: np.array(value)[masks] for key, value in colors.items()}
lrd_weights = np.array(weights)[masks]

# Define plotting parameters
gridsize = 30
extent = (8, 11.5, -2, 3)
norm = LogNorm(vmin=10**-4.5, vmax=10**0.8)

# Create the figure
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# Plot the color-color distributions
ax[0, 0].hexbin(
    colors["F200W_F277W"],
    colors["F200W_F356W"],
    C=weights,
    gridsize=gridsize,
    cmap="viridis",
    linewidth=0.2,
    reduce_C_function=np.sum,
    mincnt=1,
    norm=norm,
    extent=[-1, 2.5, -1, 4],
)
ax[0, 0].scatter(
    lrd_colors["F200W_F277W"],
    lrd_colors["F200W_F356W"],
    color="red",
    s=3,
    alpha=0.7,
)
ax[0, 0].set_xlabel("F200W - F277W")
ax[0, 0].set_ylabel("F200W - F356W")
ax[0, 0].text(
    0.05,
    0.95,
    "Red 1 (Kokorev+24)",
    transform=ax[0, 0].transAxes,
    fontsize=12,
    verticalalignment="top",
)
ax[0, 0].set_xlim(-1, 2.5)
ax[0, 0].set_ylim(-1, 4)

ax[0, 1].hexbin(
    colors["F277W_F356W"],
    colors["F277W_F444W"],
    C=weights,
    gridsize=gridsize,
    cmap="viridis",
    linewidth=0.2,
    mincnt=1,
    norm=norm,
    extent=[-2, 2.5, -1, 4],
)
ax[0, 1].scatter(
    lrd_colors["F277W_F356W"],
    lrd_colors["F277W_F444W"],
    color="red",
    s=3,
    alpha=0.7,
)
ax[0, 1].set_xlabel("F277W - F356W")
ax[0, 1].set_ylabel("F277W - F444W")
ax[0, 1].text(
    0.05,
    0.95,
    "Red 2 (Kokorev+24)",
    transform=ax[0, 1].transAxes,
    fontsize=12,
    verticalalignment="top",
)
ax[0, 1].set_xlim(-2, 2.5)
ax[0, 1].set_ylim(-1, 4)

ax[1, 0].hexbin(
    colors["F200W_F277W"],
    colors["F115W_F150W"],
    C=weights,
    gridsize=gridsize,
    cmap="viridis",
    linewidth=0.2,
    mincnt=1,
    norm=norm,
    extent=[-1, 2.5, -1, 4],
)
ax[1, 0].scatter(
    lrd_colors["F200W_F277W"],
    lrd_colors["F115W_F150W"],
    color="red",
    s=3,
    alpha=0.7,
)
ax[1, 0].text(
    0.05,
    0.95,
    "Red 1 (Kokorev+24)",
    transform=ax[1, 0].transAxes,
    fontsize=12,
    verticalalignment="top",
)
ax[1, 0].set_xlabel("F200W - F277W")
ax[1, 0].set_ylabel("F115W - F150W")
ax[1, 0].set_xlim(-1, 2.5)
ax[1, 0].set_ylim(-1, 4)

ax[1, 1].hexbin(
    colors["F277W_F356W"],
    colors["F115W_F150W"],
    C=weights,
    gridsize=gridsize,
    cmap="viridis",
    linewidth=0.2,
    mincnt=1,
    norm=norm,
    extent=[-2, 2.5, -1, 4],
)
ax[1, 1].scatter(
    lrd_colors["F277W_F356W"],
    lrd_colors["F115W_F150W"],
    color="red",
    s=3,
    alpha=0.7,
)
ax[1, 1].set_xlabel("F277W - F356W")
ax[1, 1].set_ylabel("F115W - F150W")
ax[1, 1].text(
    0.05,
    0.95,
    "Red 2 (Kokorev+24)",
    transform=ax[1, 1].transAxes,
    fontsize=12,
    verticalalignment="top",
)
ax[1, 1].set_xlim(-2, 2.5)
ax[1, 1].set_ylim(-1, 4)

# Turn on the grid for each axis
ax[0, 0].grid(True)
ax[0, 1].grid(True)
ax[1, 0].grid(True)
ax[1, 1].grid(True)

# Add a colorbar
cbar = fig.colorbar(ax[0, 0].collections[0], ax=ax)
cbar.set_label("Number of galaxies")

# Save the figure
savefig(fig, "color_distribution")
