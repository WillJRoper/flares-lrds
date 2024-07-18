"""A script for comparing the gas distributions of galaxies."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils import get_masked_synth_data, get_synth_data, savefig, SNAPSHOTS

# Define the parser
parser = argparse.ArgumentParser(description="Plot the SFZHs of galaxies.")
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

# Define plotting parameters
gridsize = 50
extent = (-2, 2, -2, 2)
norm = mcolors.LogNorm()


# Get the synthesizer data
(
    fluxes,
    colors,
    red1,
    red2,
    sizes,
    masks,
    indices,
    weights,
) = get_synth_data(data_file, "attenuated", get_weights=True)

# Get the sfzhs
gas_hmr = get_masked_synth_data(data_file, "GasHalfMassRadius")
dust_hmr = get_masked_synth_data(data_file, "DustHalfMassRadius")

# Convert from Mpc to kpc
for snap in SNAPSHOTS:
    gas_hmr[snap] *= 1e3
    dust_hmr[snap] *= 1e3

# Loop over the snapshots
for snap in SNAPSHOTS:
    # Get the data
    gas_data = gas_hmr[snap]
    dust_data = dust_hmr[snap]
    ws = weights[snap]
    mask = masks[snap]

    # Create the figure
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Draw a grid behind everything
    ax.grid(True)
    ax.set_axisbelow(True)

    # Plot the non-LRD data as a hexbin
    hb = ax.hexbin(
        gas_data[~mask],
        dust_data[~mask],
        gridsize=gridsize,
        norm=norm,
        extent=extent,
        cmap="viridis",
        linewidths=0.2,
        xscale="log",
        yscale="log",
        mincnt=ws.min(),
        C=ws[~mask],
        reduce_C_function=np.sum,
    )

    # Plot the LRD data as a scatter plot
    ax.scatter(
        gas_data[mask],
        dust_data[mask],
        c="red",
        s=1,
        alpha=0.8,
        label="LRD",
    )

    # Add the colorbar
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label(r"$\sum w_i$")

    # Set the labels
    ax.set_xlabel(r"$R_{1/2}^{\mathrm{Gas}} /$ [kpc]")
    ax.set_ylabel(r"$R_{1/2}^{\mathrm{Dust}} /$ [kpc]")

    # Draw the legend
    ax.legend()

    # Save the figure
    savefig(fig, f"gas_dust_size_{args.type}_{snap}")
