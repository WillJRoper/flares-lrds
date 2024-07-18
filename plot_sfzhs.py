"""A script for comparing average SFZHs of galaxies."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from synthesizer.grid import Grid

from utils import get_masked_synth_data, get_synth_data, savefig, SNAPSHOTS

# Define the parser
parser = argparse.ArgumentParser(description="Plot the SFZHs of galaxies.")
parser.add_argument(
    "--type",
    type=str,
    default="stellar",
    help="The type of data to plot.",
)
parser.add_argument(
    "--grid",
    type=str,
    help="The file name of the Synthesizer gird.",
)
parser.add_argument(
    "--grid-dir",
    type=str,
    help="The directory of the Synthesizer grid.",
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

# Load the grid
grid = Grid(args.grid, args.grid_dir)

# Define plotting parameters
bins = 30
extent = (
    grid.log10metallicites[0],
    grid.log10metallicites[-1],
    grid.log10ages[0],
    grid.log10ages[-1],
)
norm = mcolors.LogNorm(vmin=10**6, vmax=10**10)


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
sfzhs = get_masked_synth_data(data_file, "SFZH")

# Loop over snapshots
for snap in SNAPSHOTS:
    # Split the sfzhs into lrds and the rest
    lrd_sfzh = sfzhs[snap][masks[snap]]
    other_sfzh = sfzhs[snap][~masks[snap]]

    # Get the weighted average SFZH along the first axis (galaxy per row)
    lrd_avg_sfzh = np.average(
        lrd_sfzh, axis=0, weights=weights[snap][masks[snap]]
    )
    other_avg_sfzh = np.average(
        other_sfzh, axis=0, weights=weights[snap][~masks[snap]]
    )

    # Plot the SFZHs
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(
        lrd_avg_sfzh.T,
        extent=extent,
        aspect="auto",
        origin="lower",
        norm=norm,
        cmap="plasma",
    )
    ax[0].set_title("LRD")
    ax[0].set_xlabel(r"$\log_{10}(Z)$")
    ax[0].set_ylabel(r"$\log_{10}(\tau/\mathrm{yr})$")

    ax[1].imshow(
        other_avg_sfzh.T,
        extent=extent,
        aspect="auto",
        origin="lower",
        norm=norm,
        cmap="plasma",
    )
    ax[1].set_xlabel(r"$\log_{10}(Z)$")
    ax[1].set_ylabel(r"$\log_{10}(\tau/\mathrm{yr})$")

    # Add a colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="plasma"),
        ax=ax,
        orientation="horizontal",
    )
    cbar.set_label(r"SFZH / [$M_\odot$]")

    # Save the figure
    savefig(fig, f"sfzh_comp_{args.type}_{snap}")
