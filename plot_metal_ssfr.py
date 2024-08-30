"""A script for comparing average SFZHs of galaxies."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from utils import (
    get_synth_data,
    savefig,
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
    "--master",
    type=str,
    required=True,
    help="The master file to use.",
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
all_sfr_100 = get_master_data(
    args.master,
    indices,
    "SFR_aperture/30/100Myr",
)
all_sfr_10 = get_master_data(
    args.master,
    indices,
    "SFR_aperture/30/10Myr",
)
all_star_masses = get_master_data(
    args.master,
    indices,
    "Mstar_aperture/30",
)
all_star_metals = get_master_data(
    args.master,
    indices,
    "Metallicity/CurrentMassWeightedStellarZ",
)

# Sanitise the data by replacing 0 sfrs with a placeholder value
for snap in all_sfr_100.keys():
    all_sfr_100[snap][all_sfr_100[snap] == 0] = (
        all_star_masses[snap][all_sfr_100[snap] == 0] * 0.01
    )
    all_sfr_10[snap][all_sfr_10[snap] == 0] = (
        all_star_masses[snap][all_sfr_10[snap] == 0] * 0.01
    )

# Calculate the specific star formation rates
all_ssfr_100 = {}
for snap in all_sfr_100.keys():
    all_ssfr_100[snap] = all_sfr_100[snap] / all_star_masses[snap]

# Loop over snapshots
for snap in all_star_metals.keys():
    # Early exit if there are no galaxies in the mask
    if np.sum(masks[snap]) == 0:
        continue

    # Plot the stellar metallicity against stellar mass
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Set the grid behind the plots
    ax.grid(True)
    ax.set_axisbelow(True)

    im = ax.hexbin(
        all_star_metals[snap][~masks[snap]],
        all_ssfr_100[snap][~masks[snap]],
        C=weights[snap],
        mincnt=np.min(weights[snap]),
        linewidths=0.2,
        reduce_C_function=np.sum,
        cmap="viridis",
        gridsize=50,
        xscale="log",
        yscale="log",
        norm=mcolors.LogNorm(),
    )
    ax.scatter(
        all_star_metals[snap][masks[snap]],
        all_ssfr_100[snap][masks[snap]],
        c="red",
        label="LRDs",
        alpha=0.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$Z_{\star}$")
    ax.set_ylabel(r"$\mathrm{sSFR}_{100}$")
    ax.legend()

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\sum w_i$")

    savefig(fig, f"{snap}/stellar_metallicity_vs_ssfr_{snap}.png")
