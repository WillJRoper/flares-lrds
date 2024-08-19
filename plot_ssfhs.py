"""A script for comparing average SFHs of galaxies in terms of sSFR."""
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
mass_bins = [10**9, 10**9.5, 10**10, 10**11.5]

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

    # Extract the xs
    xs = 10 ** grid.log10ages[:-1]

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

        # Convert the SFHs to SFRs
        lrdsfh = lrd_sfh[lrd_okinds, :-1] / np.diff(10**grid.log10ages)
        othersfh = other_sfh[other_okinds, :-1] / np.diff(10**grid.log10ages)

        # Convert the SFHs in terms of SFR to sSFR by dividing by the cumalative
        # mass
        lrdsfh /= np.cumsum(lrd_sfh, axis=1)
        othersfh /= np.cumsum(other_sfh, axis=1)

        # Plot all LRDS with a low alpha
        for sfh in lrdsfh:
            ax.loglog(xs, sfh, color="red", alpha=0.05)

        # Plot the median SFHs
        if np.sum(lrd_okinds) > 0:
            ax.loglog(
                xs,
                np.median(lrdsfh, axis=0),
                label="LRD",
                color="red",
            )

        # Plot the median of the other galaxies
        if np.sum(other_okinds) > 0:
            ax.loglog(
                xs,
                np.median(othersfh, axis=0),
                label="Other",
                color="blue",
            )

        # Label the axes with the mass bin
        ax.text(
            0.95,
            0.9,
            r"$10^{"
            f"{np.log10(mass_low):.1f}"
            r"} < "
            r"M/M_\odot < 10^{"
            f"{np.log10(mass_high):.1f}"
            r"}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3", fc="w", ec="k", lw=1, alpha=0.7
            ),
        )

        # Set limtis
        ax.set_ylim(10**-1.5, 10**3.2)

    # Turn off the x-axis labels for all but the bottom plot
    for ax in axes[:-1]:
        ax.set_xticklabels([])

    # Labeled axes
    ax3.set_xlabel(r"$\mathrm{Age} / [\mathrm{yr}]$")
    ax2.set_ylabel(r"$\mathrm{SFR} / [M_\odot/\mathrm{yr}]$")

    # Add a legend
    ax3.legend(loc="lower right")

    # Save the figure
    savefig(fig, f"ssfh_{snap}_{args.type}.png")
