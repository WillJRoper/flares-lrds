"""A script for plotting histograms of galaxy properties including sSFR against stellar mass."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils import get_master_data, get_synth_data, savefig

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
parser.add_argument(
    "--spectra",
    type=str,
    nargs="+",
    default=[
        "attenuated",
    ],
    help="The spectra to plot.",
)
parser.add_argument(
    "--master",
    type=str,
    required=True,
    help="The master file to use.",
)
parser.add_argument(
    "--normalise",
    action="store_true",
    help="Normalise the histograms.",
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
bins = 30
extent = (8, 11.5, -12.5, -8.1)
norm = mcolors.LogNorm(vmin=10**-4.5, vmax=10**0.8)

# Loop over the spectra
for spectra in args.spectra:
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
    ) = get_synth_data(data_file, spectra, get_weights=True)

    # Get the master file data excluding the masks
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

    # Sanitise the data by replacing 0 sfrs with a placeholder value
    for snap in all_sfr_100.keys():
        all_sfr_100[snap][all_sfr_100[snap] == 0] = (
            all_star_masses[snap][all_sfr_100[snap] == 0] * 0.01
        )
        all_sfr_10[snap][all_sfr_10[snap] == 0] = (
            all_star_masses[snap][all_sfr_10[snap] == 0] * 0.01
        )

    # Loop over snapshots
    for snap in all_sfr_100.keys():
        # Early exit if there are no galaxies in the mask
        if np.sum(masks[snap]) == 0:
            continue

        # Calculate sSFR for 100 Myr and 10 Myr
        all_ssfr_100 = all_sfr_100[snap] / (all_star_masses[snap] * 10**10)
        all_ssfr_10 = all_sfr_10[snap] / (all_star_masses[snap] * 10**10)
        ssfr_100_lrd = all_ssfr_100[masks[snap]]
        ssfr_10_lrd = all_ssfr_10[masks[snap]]
        star_masses_lrd = all_star_masses[snap][masks[snap]] * 10**10

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        # Set the grid behind the plots
        ax1.grid(True)
        ax1.set_axisbelow(True)
        ax2.grid(True)
        ax2.set_axisbelow(True)

        # Plot for 100 Myr
        im1 = ax1.hexbin(
            all_star_masses[snap][~masks[snap]] * 10**10,
            all_ssfr_100[~masks[snap]],
            C=weights[snap][~masks[snap]],
            mincnt=np.min(weights[snap][~masks[snap]]),
            linewidths=0.2,
            reduce_C_function=np.sum,
            cmap="viridis",
            gridsize=50,
            xscale="log",
            yscale="log",
            norm=norm,
            extent=extent,
        )
        ax1.scatter(
            star_masses_lrd,
            ssfr_100_lrd,
            c="red",
            label="LRDs",
            alpha=0.5,
        )
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel(r"$M_{\star} \, [M_\odot]$")
        ax1.set_ylabel(r"$\mathrm{sSFR}_{100} \, [\mathrm{yr}^{-1}]$")

        # Plot for 10 Myr
        im2 = ax2.hexbin(
            all_star_masses[snap][~masks[snap]] * 10**10,
            all_ssfr_10[~masks[snap]],
            C=weights[snap][~masks[snap]],
            mincnt=np.min(weights[snap][~masks[snap]]),
            linewidths=0.2,
            reduce_C_function=np.sum,
            cmap="viridis",
            gridsize=50,
            xscale="log",
            yscale="log",
            norm=norm,
            extent=extent,
        )
        ax2.scatter(
            star_masses_lrd,
            ssfr_10_lrd,
            c="red",
            label="LRDs",
            alpha=0.5,
        )
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"$M_{\star} \, [M_\odot]$")
        ax2.set_ylabel(r"$\mathrm{sSFR}_{10} \, [\mathrm{yr}^{-1}]$")
        ax2.legend()
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar2.set_label("$\sum w_i$")

        # Set the y limits so the plots match
        ax1.set_ylim(ax2.get_ylim())

        # Save the figure
        savefig(fig, f"{spectra}/{snap}/ssfr_vs_stellar_mass_{snap}.png")
