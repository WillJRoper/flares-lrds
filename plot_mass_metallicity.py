"""A script for plotting histograms of galaxy properties including stellar metallicity against stellar mass."""
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
    all_star_metals = get_master_data(
        args.master,
        indices,
        "Metallicity/CurrentMassWeightedStellarZ",
    )
    all_star_masses = get_master_data(
        args.master,
        indices,
        "Mstar_aperture/30",
    )

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
            all_star_masses[snap][~masks[snap]] * 10**10,
            all_star_metals[snap][~masks[snap]],
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
            all_star_masses[snap][masks[snap]] * 10**10,
            all_star_metals[snap][masks[snap]],
            c="red",
            label="LRDs",
            alpha=0.5,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$M_{\star} \, [M_\odot]$")
        ax.set_ylabel(r"$\log_{10}(Z_{\star})$")
        ax.legend()

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("$\sum w_i$")

        savefig(
            fig, f"{spectra}/{snap}/stellar_metallicity_vs_mass_{snap}.png"
        )
