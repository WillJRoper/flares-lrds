"""A script to plot the UV and Optical slopes from FLARES."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
import argparse
from utils import REGIONS, SNAPSHOTS, get_fluxes_colors

# Define the command line arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--type",
    type=str,
    default="stellar",
    help="The type of data to plot",
)

args = parser.parse_args()

# Define the data file
if args.type == "stellar":
    data_file = "data/pure_stellar_<region>_<snap>.hdf5"
elif args.type == "agn":
    data_file = "data/pure_agn_<region>_<snap>.hdf5"
else:
    data_file = "data/combined_<region>_<snap>.hdf5"

# Get regions and snapshots
regions = REGIONS
snaps = SNAPSHOTS

# Define the spectra keys we'll read
spectra_keys = ["attenuated", "reprocessed"]

# Get the fluxes, colors and masks
att_fluxes, att_colors, att_red1, att_red2 = get_fluxes_colors(
    data_file, "attenuated"
)
rep_fluxes, rep_colors, rep_red1, rep_red2 = get_fluxes_colors(
    data_file, "reprocessed"
)

# Combine masks
att_red = np.logical_or(att_red1, att_red2)
rep_red = np.logical_or(rep_red1, rep_red2)

# Get the slopes
uv_slopes = {}
optical_slopes = {}
compacts = {}
# Loop over regions and snapshots getting the data
for reg in regions:
    for snap in snaps:
        with h5py.File(
            data_file.replace("<region>", reg).replace("<snap>", snap), "r"
        ) as hdf:
            for spec in spectra_keys:
                try:
                    uv_slopes.setdefault(spec, {}).setdefault(snap, []).extend(
                        hdf["UVSlopes"][spec][...]
                    )
                    optical_slopes.setdefault(spec, {}).setdefault(
                        snap, []
                    ).extend(hdf["OpticalSlopes"][spec][...])
                except KeyError as e:
                    print(e)
                except TypeError as e:
                    print(e)

# Convert slopes to arrays
for spec in spectra_keys:
    for snap in snaps:
        uv_slopes[spec][snap] = np.array(uv_slopes[spec][snap])
        optical_slopes[spec][snap] = np.array(optical_slopes[spec][snap])

# Define plotting parameters
gridsize = 50
norm = mcolors.LogNorm(1, 10**4)
extent = (-2.7, 0, -2.7, 0)

# Loop over the snapshots
for snap in snaps:
    # Plot hexbins of slope vs slope for all galaxies and the red sample
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))

    axs[0].hexbin(
        optical_slopes["attenuated"][snap],
        uv_slopes["attenuated"][snap],
        gridsize=gridsize,
        cmap="viridis",
        norm=norm,
        extent=extent,
        mincnt=1,
        linewidth=0.2,
    )
    axs[0].text(
        0.95,
        0.05,
        "All Galaxies",
        ha="right",
        va="bottom",
        transform=axs[0].transAxes,
        fontsize=12,
        color="k",
    )

    axs[1].hexbin(
        optical_slopes["attenuated"][snap][att_red],
        uv_slopes["attenuated"][snap][att_red],
        gridsize=gridsize,
        cmap="viridis",
        norm=norm,
        extent=extent,
        mincnt=1,
        linewidth=0.2,
    )
    axs[1].text(
        0.95,
        0.05,
        "(Red 1 | Red 2) (Kokorev+24)",
        ha="right",
        va="bottom",
        transform=axs[1].transAxes,
        fontsize=12,
        color="k",
    )

    # Label the axes
    axs[0].set_xlabel("Optical Slope")
    axs[0].set_ylabel("UV Slope")
    axs[1].set_xlabel("Optical Slope")

    # Draw a colorbar on the right
    cbar = fig.colorbar(
        mappable=axs[1].collections[0], ax=axs, orientation="vertical"
    )
    cbar.set_label("$N$")

    fig.savefig(f"slopes_attenuated_{snap}", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Plot hexbins of slope vs slope for all galaxies and the red sample
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))

    axs[0].hexbin(
        optical_slopes["reprocessed"][snap],
        uv_slopes["reprocessed"][snap],
        gridsize=gridsize,
        cmap="viridis",
        norm=norm,
        extent=extent,
        mincnt=1,
        linewidth=0.2,
    )
    axs[0].text(
        0.95,
        0.05,
        "All Galaxies",
        ha="right",
        va="bottom",
        transform=axs[0].transAxes,
        fontsize=12,
        color="k",
    )

    axs[1].hexbin(
        optical_slopes["reprocessed"][snap][rep_red],
        uv_slopes["reprocessed"][snap][rep_red],
        gridsize=gridsize,
        cmap="viridis",
        norm=norm,
        extent=extent,
        mincnt=1,
        linewidth=0.2,
    )
    axs[1].text(
        0.95,
        0.05,
        "(Red 1 | Red 2) (Kokorev+24)",
        ha="right",
        va="bottom",
        transform=axs[1].transAxes,
        fontsize=12,
        color="k",
    )

    # Label the axes
    axs[0].set_xlabel("Optical Slope")
    axs[0].set_ylabel("UV Slope")
    axs[1].set_xlabel("Optical Slope")

    # Draw a colorbar on the right
    cbar = fig.colorbar(
        mappable=axs[1].collections[0], ax=axs, orientation="vertical"
    )
    cbar.set_label("$N$")

    fig.savefig(f"slopes_reprocessed_{snap}", dpi=100, bbox_inches="tight")
    plt.close(fig)
