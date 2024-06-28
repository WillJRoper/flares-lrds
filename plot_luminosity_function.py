"""A script for plotting the luminsoity function of different galaxy types."""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import h5py
from unyt import angstrom
from astropy.cosmology import Planck15 as cosmo

from synthesizer.conversions import lnu_to_absolute_mag, fnu_to_lnu

from utils import get_synth_data, SNAPSHOTS
from synthesize_flares import get_flares_filters

# Define the parser
parser = argparse.ArgumentParser(
    description="Plot the luminosity function of galaxies."
)
parser.add_argument(
    "--type",
    type=str,
    default="stellar",
    help="The type of data to plot.",
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

# Define the filters we'll use
filters = get_flares_filters("lrd_filters.hdf5")

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
    weights,
) = get_synth_data(data_file, "attenuated", get_weights=True)
(
    rep_fluxes,
    rep_colors,
    rep_red1,
    rep_red2,
    rep_sizes,
    rep_masks,
    _,
) = get_synth_data(data_file, "reprocessed")

# Define magnitude bins
bins = np.arange(-27, -19.5, 0.5)
bin_cents = (bins[:-1] + bins[1:]) / 2

# Define the volume
volume = 3200**3  # Mpc^3

# Loop over the spectra keys
for spec in spectra_keys:
    # Loop over snapshots
    for snap in SNAPSHOTS:
        # Get the redshift from the snap tag
        z = float(snap.split("_")[-1].replace("z", "").replace("p", "."))

        # Which filter contains the rest frame UV?
        uv_filter = filters.find_filter(
            1500 * angstrom, redshift=z, method="transmission"
        )

        # Get the flux for this filter
        fluxes = (
            att_fluxes[snap][uv_filter.filter_code]
            if spec == "attenuated"
            else rep_fluxes[snap][uv_filter.filter_code]
        )

        # Get the right mask
        mask = att_masks[snap] if spec == "attenuated" else rep_masks[snap]

        # Convert flux to absolute magnitude
        mags = lnu_to_absolute_mag(fnu_to_lnu(fluxes, cosmo, z))

        # Compute the luminosity function full
        hist, _ = np.histogram(mags, bins=bins, weights=weights[snap])
        phi = hist / volume / np.diff(bins)

        # Compute the LRD luminosity function (masked LF)
        hist, _ = np.histogram(
            mags[mask], bins=bins, weights=weights[snap][mask]
        )
        lrd_phi = hist / volume / np.diff(bins)

        # Plot the luminosity function
        fig, ax = plt.subplots()

        ax.scatter(
            bin_cents,
            phi,
            label="All",
            color="grey",
            alpha=0.8,
            marker="s",
        )
        ax.scatter(
            bin_cents,
            lrd_phi,
            label="LRD",
            color="red",
            alpha=0.8,
            marker="o",
        )

        ax.set_yscale("log")
        ax.set_xlabel("$M_{1500}$")

        ax.set_ylabel(r"$\phi$ / [Mpc$^{-3}$ mag$^{-1}$]")

        ax.legend()

        fig.savefig(f"plots/luminosity_function_{spec}_{snap}.png")
        plt.close(fig)
