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

# Define plotting parameters
bins = 30
extent = (
    grid.log10metallicities[0],
    grid.log10metallicities[-1],
    grid.log10ages[0],
    grid.log10ages[-1],
)

# Loop over snapshots
for snap in SNAPSHOTS:
    # Split the sfzhs into lrds and the rest
    lrd_sfzh = sfzhs[snap][masks[snap]]
    other_sfzh = sfzhs[snap][~masks[snap]]

    # Skip if there are no LRD galaxies
    if np.all(~masks[snap]):
        continue

    # Get the weighted average SFZH along the first axis (galaxy per row)
    lrd_avg_sfzh = np.average(
        lrd_sfzh, axis=0, weights=weights[snap][masks[snap]]
    )
    other_avg_sfzh = np.average(
        other_sfzh, axis=0, weights=weights[snap][~masks[snap]]
    )

    # Make the norm
    norm = mcolors.LogNorm(
        vmin=10**6,
        vmax=np.max((np.max(lrd_avg_sfzh), np.max(other_avg_sfzh))),
    )

    # Plot the SFZHs as a pcolormesh with a residual image
    fig = plt.figure(figsize=(3.5 * 3, 3.5 * 1.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    ax = [
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1]),
        fig.add_subplot(gs[2]),
    ]

    ax[0].pcolormesh(
        grid.log10ages,
        grid.log10metallicities,
        lrd_avg_sfzh.T,
        aspect="auto",
        origin="lower",
        norm=norm,
        cmap="plasma",
        shading="nearest",
    )
    ax[0].set_title("LRD")
    ax[0].set_ylabel(r"$\log_{10}(Z)$")
    ax[0].set_xlabel(r"$\log_{10}(\tau/\mathrm{yr})$")

    ax[1].pcolormesh(
        grid.log10ages,
        grid.log10metallicities,
        other_avg_sfzh.T,
        aspect="auto",
        origin="lower",
        norm=norm,
        cmap="plasma",
        shading="nearest",
    )
    ax[1].set_ylabel(r"$\log_{10}(Z)$")
    ax[1].set_xlabel(r"$\log_{10}(\tau/\mathrm{yr})$")

    resi = ax[2].pcolormesh(
        grid.log10ages,
        grid.log10metallicities,
        lrd_avg_sfzh.T - other_avg_sfzh.T,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        shading="nearest",
    )
    ax[2].set_ylabel(r"$\log_{10}(Z)$")
    ax[2].set_xlabel(r"$\log_{10}(\tau/\mathrm{yr})$")

    # Add a colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="plasma"),
    )
    cbar.set_label(r"SFZH / [$M_\odot$]")

    # Add a second colorbar for the residuals
    cbar2 = fig.colorbar(resi, ax=ax[2])
    cbar2.set_label(r"LRD - Other")

    # Save the figure
    savefig(fig, f"sfzh_comp_{args.type}_{snap}")
