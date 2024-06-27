"""A script to plot the UV and Optical slopes from FLARES."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
import argparse
from utils import REGIONS, SNAPSHOTS, get_synth_data

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
(
    att_fluxes,
    att_colors,
    att_red1,
    att_red2,
    att_sizes,
    att_masks,
    _,
) = get_synth_data(data_file, "attenuated")
(
    rep_fluxes,
    rep_colors,
    rep_red1,
    rep_red2,
    rep_sizes,
    rep_masks,
    _,
) = get_synth_data(data_file, "reprocessed")

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
norm = mcolors.LogNorm(1, 10**3.5, clip=True)
extent = (-2.7, 0, -2.7, 0)
extent2 = (0, 4.3, 0.7, 2)

# Loop over the snapshots
for snap in snaps:
    if len(uv_slopes["attenuated"][snap]) == 0:
        continue

    print(f"At {snap} have {np.sum(att_masks[snap])} LRDs")

    # Plot hexbins of slope vs slope for all galaxies and the red sample
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))

    # Turn on a grid and make sure it is behind everything
    axs[0].grid(True)
    axs[0].set_axisbelow(True)
    axs[1].grid(True)
    axs[1].set_axisbelow(True)

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
        fontsize=8,
        color="k",
        bbox=dict(
            boxstyle="round,pad=0.3", fc="grey", ec="w", lw=1, alpha=0.7
        ),
    )

    axs[1].hexbin(
        optical_slopes["attenuated"][snap][att_masks[snap]],
        uv_slopes["attenuated"][snap][att_masks[snap]],
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
        "Red 1 | Red 2 (Kokorev+24)",
        ha="right",
        va="bottom",
        transform=axs[1].transAxes,
        fontsize=8,
        color="k",
        bbox=dict(
            boxstyle="round,pad=0.3", fc="grey", ec="w", lw=1, alpha=0.7
        ),
    )

    # Turn off the second y axis
    axs[1].set_yticklabels([])

    # Label the axes
    axs[0].set_xlabel("Optical Slope")
    axs[0].set_ylabel("UV Slope")
    axs[1].set_xlabel("Optical Slope")

    # Draw a colorbar on the right
    cbar = fig.colorbar(
        mappable=axs[1].collections[0], ax=axs, orientation="vertical"
    )
    cbar.set_label("$N$")

    fig.savefig(
        f"plots/slopes_attenuated_{snap}", dpi=100, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot hexbins of slope vs slope for all galaxies and the red sample
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))

    # Turn on a grid and make sure it is behind everything
    axs[0].grid(True)
    axs[0].set_axisbelow(True)
    axs[1].grid(True)
    axs[1].set_axisbelow(True)

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
        fontsize=8,
        color="k",
        bbox=dict(
            boxstyle="round,pad=0.3", fc="grey", ec="w", lw=1, alpha=0.7
        ),
    )

    axs[1].hexbin(
        optical_slopes["reprocessed"][snap][rep_masks[snap]],
        uv_slopes["reprocessed"][snap][rep_masks[snap]],
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
        "Red 1 | Red 2 (Kokorev+24)",
        ha="right",
        va="bottom",
        transform=axs[1].transAxes,
        fontsize=8,
        color="k",
        bbox=dict(
            boxstyle="round,pad=0.3", fc="grey", ec="w", lw=1, alpha=0.7
        ),
    )

    # Turn off the second y axis
    axs[1].set_yticklabels([])

    # Label the axes
    axs[0].set_xlabel("Optical Slope")
    axs[0].set_ylabel("UV Slope")
    axs[1].set_xlabel("Optical Slope")

    # Draw a colorbar on the right
    cbar = fig.colorbar(
        mappable=axs[1].collections[0], ax=axs, orientation="vertical"
    )
    cbar.set_label("$N$")

    fig.savefig(
        f"plots/slopes_reprocessed_{snap}", dpi=100, bbox_inches="tight"
    )
    plt.close(fig)

    okinds = att_fluxes[snap]["F444W"] > 0

    # Plot the ratio between slopes against the F444W flux
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))

    # Turn on a grid and make sure it is behind everything
    axs[0].grid(True)
    axs[0].set_axisbelow(True)
    axs[1].grid(True)
    axs[1].set_axisbelow(True)

    axs[0].hexbin(
        att_fluxes[snap]["F444W"][okinds],
        optical_slopes["attenuated"][snap][okinds]
        / uv_slopes["attenuated"][snap][okinds],
        gridsize=gridsize,
        cmap="viridis",
        norm=norm,
        mincnt=1,
        linewidth=0.2,
        xscale="log",
        extent=extent2,
    )
    axs[0].text(
        0.95,
        0.95,
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

    axs[1].hexbin(
        att_fluxes[snap]["F444W"][np.logical_and(att_masks[snap], okinds)],
        optical_slopes["attenuated"][snap][
            np.logical_and(att_masks[snap], okinds)
        ]
        / uv_slopes["attenuated"][snap][
            np.logical_and(att_masks[snap], okinds)
        ],
        gridsize=gridsize,
        cmap="viridis",
        norm=norm,
        mincnt=1,
        linewidth=0.2,
        xscale="log",
        extent=extent2,
    )

    axs[1].text(
        0.95,
        0.95,
        "Red 1 | Red 2 (Kokorev+24)",
        ha="right",
        va="bottom",
        transform=axs[1].transAxes,
        fontsize=8,
        color="k",
        bbox=dict(
            boxstyle="round,pad=0.3", fc="grey", ec="w", lw=1, alpha=0.7
        ),
    )

    # Turn off the second y axis
    axs[1].set_yticklabels([])

    # Label the axes
    axs[0].set_xlabel("F444W Flux")
    axs[0].set_ylabel("Optical Slope / UV Slope")
    axs[1].set_xlabel("F444W Flux")

    # Draw a colorbar on the right
    cbar = fig.colorbar(
        mappable=axs[1].collections[0], ax=axs, orientation="vertical"
    )
    cbar.set_label("$N$")

    fig.savefig(
        f"plots/slopes_ratio_attenuated_{snap}",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Plot the ratio between slopes against the F444W flux
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))

    # Turn on a grid and make sure it is behind everything
    axs[0].grid(True)
    axs[0].set_axisbelow(True)
    axs[1].grid(True)
    axs[1].set_axisbelow(True)

    axs[0].hexbin(
        rep_fluxes[snap]["F444W"][okinds],
        optical_slopes["reprocessed"][snap][okinds]
        / uv_slopes["reprocessed"][snap][okinds],
        gridsize=gridsize,
        cmap="viridis",
        norm=norm,
        mincnt=1,
        linewidth=0.2,
        xscale="log",
        extent=extent2,
    )
    axs[0].text(
        0.95,
        0.95,
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

    axs[1].hexbin(
        rep_fluxes[snap]["F444W"][np.logical_and(rep_masks[snap], okinds)],
        optical_slopes["reprocessed"][snap][
            np.logical_and(rep_masks[snap], okinds)
        ]
        / uv_slopes["reprocessed"][snap][
            np.logical_and(rep_masks[snap], okinds)
        ],
        gridsize=gridsize,
        cmap="viridis",
        norm=norm,
        mincnt=1,
        linewidth=0.2,
        xscale="log",
        extent=extent2,
    )

    axs[1].text(
        0.95,
        0.95,
        "Red 1 | Red 2 (Kokorev+24)",
        ha="right",
        va="bottom",
        transform=axs[1].transAxes,
        fontsize=8,
        color="k",
        bbox=dict(
            boxstyle="round,pad=0.3", fc="grey", ec="w", lw=1, alpha=0.7
        ),
    )

    # Turn off the second y axis
    axs[1].set_yticklabels([])

    # Label the axes
    axs[0].set_xlabel("F444W Flux")
    axs[0].set_ylabel("Optical Slope / UV Slope")
    axs[1].set_xlabel("F444W Flux")

    # Draw a colorbar on the right
    cbar = fig.colorbar(
        mappable=axs[1].collections[0], ax=axs, orientation="vertical"
    )
    cbar.set_label("$N$")

    fig.savefig(
        f"plots/slopes_ratio_reprocessed_{snap}",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

    extent2 = (-1.1, 1.3, 0.7, 2)

    # Plot the ratio between slopes against the F444W flux
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))

    # Turn on a grid and make sure it is behind everything
    axs[0].grid(True)
    axs[0].set_axisbelow(True)
    axs[1].grid(True)
    axs[1].set_axisbelow(True)

    axs[0].hexbin(
        att_sizes[snap]["F444W"][okinds],
        optical_slopes["attenuated"][snap][okinds]
        / uv_slopes["attenuated"][snap][okinds],
        gridsize=gridsize,
        cmap="viridis",
        norm=norm,
        mincnt=1,
        linewidth=0.2,
        xscale="log",
        extent=extent2,
    )
    axs[0].text(
        0.95,
        0.95,
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

    axs[1].hexbin(
        att_sizes[snap]["F444W"][np.logical_and(att_masks[snap], okinds)],
        optical_slopes["attenuated"][snap][
            np.logical_and(att_masks[snap], okinds)
        ]
        / uv_slopes["attenuated"][snap][
            np.logical_and(att_masks[snap], okinds)
        ],
        gridsize=gridsize,
        cmap="viridis",
        norm=norm,
        mincnt=1,
        linewidth=0.2,
        xscale="log",
        extent=extent2,
    )

    axs[1].text(
        0.95,
        0.95,
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

    # Turn off the second y axis
    axs[1].set_yticklabels([])

    # Label the axes
    axs[0].set_xlabel("$R_{1/2}$ / [kpc]")
    axs[0].set_ylabel("Optical Slope / UV Slope")
    axs[1].set_xlabel("$R_{1/2}$ / [kpc]")

    # Draw a colorbar on the right
    cbar = fig.colorbar(
        mappable=axs[1].collections[0], ax=axs, orientation="vertical"
    )
    cbar.set_label("$N$")

    fig.savefig(
        f"plots/slopes_ratio_attenuated_size_{snap}",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)
