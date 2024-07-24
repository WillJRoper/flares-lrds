"""A script for comparing average SFZHs of galaxies."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from synthesizer.grid import Grid

from utils import (
    get_masked_synth_data,
    get_synth_data,
    savefig,
    SNAPSHOTS,
    get_master_data,
)

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
    "--master",
    type=str,
    required=True,
    help="The master file to use.",
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
star_masses = get_master_data(
    args.master,
    indices,
    "Mstar_aperture/30",
)


# Define plotting parameters
bins = 30
extent = (
    grid.log10metallicities[0],
    grid.log10metallicities[-1],
    grid.log10ages[0],
    grid.log10ages[-1],
)

# Define mass bins
mass_bins = np.logspace(9, 11.5, 4)

# Loop over snapshots
for snap in SNAPSHOTS:
    # Split the sfzhs into lrds and the rest
    lrd_sfzh = sfzhs[snap][masks[snap]]
    other_sfzh = sfzhs[snap][~masks[snap]]
    lrd_masses = star_masses[snap][masks[snap]] * 10**10
    other_masses = star_masses[snap][~masks[snap]] * 10**10

    # Skip if there are no LRD galaxies
    if len(lrd_sfzh) == 0 or len(other_sfzh) == 0:
        continue

    # Sum over the metallicity axis of the sfzhs to get the SFH
    lrd_sfh = np.sum(lrd_sfzh, axis=2)
    other_sfh = np.sum(other_sfzh, axis=2)
    print(lrd_sfh.shape)
    print(np.median(lrd_sfh, axis=0).shape)

    # Create the figure
    fig = plt.figure(figsize=(3.5, 3 * 3.5))
    gs = fig.add_gridspec(3, 1, hspace=0.0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    axes = [ax1, ax2, ax3]
    for ax in axes:
        ax.grid(True)
        ax.set_axisbelow(True)

    # Loop over axes and mass bins
    for ax, (mass_low, mass_high) in zip(
        axes, zip(mass_bins[:-1], mass_bins[1:])
    ):
        # Get the indices of the galaxies in the mass bin
        lrd_okinds = np.where(
            (lrd_masses > mass_low) & (lrd_masses < mass_high)
        )[0]
        other_okinds = np.where(
            (other_masses > mass_low) & (other_masses < mass_high)
        )[0]

        # Plot all LRDS with a low alpha
        for sfzh in lrd_sfzh[lrd_okinds]:
            ax.semilogy(
                grid.log10ages, np.sum(sfzh, axis=1), color="red", alpha=0.1
            )

        # Plot the median SFHs
        if np.sum(lrd_okinds) > 0:
            ax.semilogy(
                grid.log10ages,
                np.median(lrd_sfh[lrd_okinds], axis=0),
                label="LRD",
                color="red",
            )

        # Plot the median of the other galaxies
        if np.sum(other_okinds) > 0:
            ax.semilogy(
                grid.log10ages,
                np.median(other_sfh[other_okinds], axis=0),
                label="Other",
                color="blue",
            )

        # Set limtis
        ax.set_ylim(10**5, 10**10.2)

    # Turn off the x-axis labels for all but the bottom plot
    for ax in axes[:-1]:
        ax.set_xticklabels([])

    # Labeled axes
    ax3.set_xlabel("log10(Age)")
    ax2.set_ylabel("SFH")

    # Add a legend
    ax1.legend()

    # Save the figure
    savefig(fig, f"sfh_{snap}_{args.type}.png")
