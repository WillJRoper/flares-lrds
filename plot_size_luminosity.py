"""A script to plot the size-luminosity relation of the galaxies."""
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from utils import get_synth_data, SNAPSHOTS, FILTER_CODES

# Set up the argument parser
parser = argparse.ArgumentParser(
    description="Plot the size-luminosity relation of the galaxies."
)

# Add the arguments
parser.add_argument(
    "--type", type=str, default="stellar", help="The type of data to plot."
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

# Define the spectra keys we'll read
spectra_keys = ["attenuated", "reprocessed"]

# Get the fluxes, colors and masks
(
    att_fluxes,
    att_colors,
    att_red1,
    att_red2,
    att_sizes,
    att_masks,
) = get_synth_data(data_file, "attenuated")
(
    rep_fluxes,
    rep_colors,
    rep_red1,
    rep_red2,
    rep_sizes,
    rep_masks,
) = get_synth_data(data_file, "reprocessed")

# Define plotting parameters
gridsize = 50
norm = mcolors.LogNorm(1, 10**3.5, clip=True)
extent2 = (0, 4.3, -2, 1)

# Loop over snapshots
for snap in SNAPSHOTS:
    for filt in FILTER_CODES:
        filt = filt.split(".")[-1]

        # Plot the size-luminosity relation
        fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))

        # Draw the grid and make sure its in the background
        axs[0].set_axisbelow(True)
        axs[1].set_axisbelow(True)
        axs[0].grid(True)
        axs[1].grid(True)

        # Plot the size-luminosity relation with no mask
        axs[0].hexbin(
            att_fluxes[snap][filt],
            att_sizes[snap][filt],
            gridsize=gridsize,
            norm=norm,
            extent=extent2,
            cmap="viridis",
            linewidths=0.2,
            xscale="log",
            yscale="log",
        )
        axs[0].text(
            0.95,
            0.05,
            "All Galaxies",
            ha="right",
            va="bottom",
            transform=axs[0].transAxes,
            fontsize=8,
            color="k",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="grey", ec="w", lw=1, alpha=0.7
            ),
        )

        # Plot the size-luminosity relation with the mask
        axs[1].hexbin(
            att_fluxes[snap][filt][att_masks[snap]],
            att_sizes[snap][filt][att_masks[snap]],
            gridsize=gridsize,
            norm=norm,
            extent=extent2,
            cmap="viridis",
            linewidths=0.2,
            xscale="log",
            yscale="log",
        )
        axs[1].text(
            0.95,
            0.05,
            "LRDs",
            ha="right",
            va="bottom",
            transform=axs[1].transAxes,
            fontsize=8,
            color="k",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="grey", ec="w", lw=1, alpha=0.7
            ),
        )

        # Set the labels
        axs[0].set_xlabel("Flux / [nJy]")
        axs[0].set_ylabel("$R_{1/2}$ / [kpc]")
        axs[1].set_xlabel("Flux / [nJy]")

        # Remove the y-axis label from the right plot
        axs[1].set_yticklabels([])

        # Draw the colorbar
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
            ax=axs,
            orientation="vertical",
            pad=0.05,
            aspect=30,
        )
        cb.set_label("$N$")

        # Save the figure
        fig.savefig(
            f"plots/size_luminosity_attenuated_{snap}_{filt}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot the size-luminosity relation
        fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))

        # Draw the grid and make sure its in the background
        axs[0].set_axisbelow(True)
        axs[1].set_axisbelow(True)
        axs[0].grid(True)
        axs[1].grid(True)

        # Plot the size-luminosity relation with no mask
        axs[0].hexbin(
            rep_fluxes[snap][filt],
            rep_sizes[snap][filt],
            gridsize=gridsize,
            norm=norm,
            extent=extent2,
            cmap="viridis",
            linewidths=0.2,
            xscale="log",
            yscale="log",
        )
        axs[0].text(
            0.95,
            0.05,
            "All Galaxies",
            ha="right",
            va="bottom",
            transform=axs[0].transAxes,
            fontsize=8,
            color="k",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="grey", ec="w", lw=1, alpha=0.7
            ),
        )

        # Plot the size-luminosity relation with the mask
        axs[1].hexbin(
            rep_fluxes[snap][filt][rep_masks[snap]],
            rep_sizes[snap][filt][rep_masks[snap]],
            gridsize=gridsize,
            norm=norm,
            extent=extent2,
            cmap="viridis",
            linewidths=0.2,
            xscale="log",
            yscale="log",
        )
        axs[1].text(
            0.95,
            0.05,
            "LRDs",
            ha="right",
            va="bottom",
            transform=axs[1].transAxes,
            fontsize=8,
            color="k",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="grey", ec="w", lw=1, alpha=0.7
            ),
        )

        # Set the labels
        axs[0].set_xlabel("Flux / [nJy]")
        axs[0].set_ylabel("$R_{1/2}$ / [kpc]")
        axs[1].set_xlabel("Flux / [nJy]")

        # Remove the y-axis label from the right plot
        axs[1].set_yticklabels([])

        # Draw the colorbar
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
            ax=axs,
            orientation="vertical",
            pad=0.05,
            aspect=30,
        )
        cb.set_label("$N$")

        # Save the figure
        fig.savefig(
            f"plots/size_luminosity_reprocessed_{snap}_{filt}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
