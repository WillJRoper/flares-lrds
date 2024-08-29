"""A script for plotting the luminsoity function of different galaxy types."""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import h5py
from unyt import angstrom
from astropy.cosmology import Planck15 as cosmo

from synthesizer.conversions import lnu_to_absolute_mag, fnu_to_lnu
from synthesizer.units import default_units

from utils import get_synth_data, SNAPSHOTS, FILTER_CODES, savefig
from synthesize_flares import get_flares_filters

# Define the parser
parser = argparse.ArgumentParser(
    description="Plot the luminosity function of galaxies."
)
parser.add_argument(
    "--spec-type",
    type=str,
    default="attenuated",
    help="The type of data to plot.",
)

# Parse the arguments
args = parser.parse_args()

# Define the data file
data_file = "data/combined_<region>_<snap>.hdf5"

# Define the filters we'll use
filters = get_flares_filters("lrd_filters.hdf5")

# Get the fluxes, colors and masks
(
    fluxes,
    colors,
    red1,
    red2,
    sizes,
    masks,
    _,
    weights,
) = get_synth_data(data_file, args.spec_type, get_weights=True)

# Define magnitude bins
bins = np.arange(-25, -15.5, 1)
bin_cents = (bins[:-1] + bins[1:]) / 2

# Define the volume
volume = 3200**3  # Mpc^3

# Loop over snapshots
for snap in SNAPSHOTS:
    # Get the redshift from the snap tag
    z = float(snap.split("_")[-1].replace("z", "").replace("p", "."))

    # Find the filter containing 1500 Angstrom
    filt = filters.find_filter(
        2000,
        redshift=z,
        method="transmission",
    )

    # Get the flux for this filter
    flux = fluxes[snap][filt.filter_code.split(".")[-1]]

    # Get the right mask
    mask = masks[snap]

    # Convert flux to absolute magnitude
    mags = lnu_to_absolute_mag(fnu_to_lnu(flux, cosmo, z))

    # Compute the luminosity function full
    hist, _ = np.histogram(mags, bins=bins)
    whist, _ = np.histogram(mags, bins=bins, weights=weights[snap])
    hist = hist.astype(float)
    whist = whist.astype(float)
    hist *= whist
    phi = hist / volume / np.diff(bins)

    # Compute the LRD luminosity function (masked LF)
    hist, _ = np.histogram(mags[mask], bins=bins)
    whist, _ = np.histogram(mags[mask], bins=bins, weights=weights[snap][mask])
    hist = hist.astype(float)
    whist = whist.astype(float)
    hist *= whist
    lrd_phi = hist / volume / np.diff(bins)

    # Plot the luminosity function
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)

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

    # Reverse the x axis
    ax.set_xlim(ax.get_xlim()[::-1])

    ax.set_yscale("log")
    ax.set_xlabel("$M_{2000}$")

    ax.set_ylabel(r"$\phi$ / [Mpc$^{-3}$ mag$^{-1}$]")

    ax.legend()

    savefig(
        fig,
        f"UVLF/luminosity_function_{args.spec_type}_"
        f"{snap}_{filt.filter_code.replace('/', '')}",
    )


# Define magnitude bins
bins = np.logspace(0, np.log10(2000), 20)
bin_cents = (bins[:-1] + bins[1:]) / 2

# Define the volume
volume = 3200**3  # Mpc^3

# Loop over snapshots
for snap in SNAPSHOTS:
    # Get the redshift from the snap tag
    z = float(snap.split("_")[-1].replace("z", "").replace("p", "."))

    # Find the filter containing 1500 Angstrom
    filt = filters.find_filter(
        2000,
        redshift=z,
        method="transmission",
    )

    # Get the flux for this filter
    flux = fluxes[snap][filt.filter_code.split(".")[-1]]

    # Get the right mask
    mask = masks[snap]

    # Convert flux to absolute magnitude
    mags = flux

    # Compute the luminosity function full
    hist, _ = np.histogram(mags, bins=bins)
    whist, _ = np.histogram(mags, bins=bins, weights=weights[snap])
    hist = hist.astype(float)
    whist = whist.astype(float)
    hist *= whist
    phi = hist / volume / np.diff(bins)

    # Compute the LRD luminosity function (masked LF)
    hist, _ = np.histogram(mags[mask], bins=bins)
    whist, _ = np.histogram(mags[mask], bins=bins, weights=weights[snap][mask])
    hist = hist.astype(float)
    whist = whist.astype(float)
    hist *= whist
    lrd_phi = hist / volume / np.diff(bins)

    # Plot the luminosity function
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.semilogx()

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
    ax.set_xlabel("$F_{2000} / [nJy]$")

    ax.set_ylabel(r"$\phi$ / [Mpc$^{-3}$ dex$^{-1}$]")

    ax.legend()

    savefig(
        fig,
        f"UVLF/luminosity_function_nomag_{args.spec_type}_"
        f"{snap}_{filt.filter_code.replace('/', '')}",
    )
