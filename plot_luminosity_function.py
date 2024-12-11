"""A script for plotting the luminsoity function of different galaxy types."""

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from synthesizer.conversions import lnu_to_absolute_mag

from utils import SNAPSHOTS, savefig

# Define the parser
parser = argparse.ArgumentParser(
    description="Plot the luminosity function of galaxies."
)
parser.add_argument(
    "--spec-type",
    type=str,
    default="total",
    help="The type of data to plot.",
)

# Parse the arguments
args = parser.parse_args()

# Define the data file
data_file = "data/combined_<snap>.hdf5"

# Define magnitude bins
bins = np.arange(-25, -15.5, 1)
bin_cents = (bins[:-1] + bins[1:]) / 2

# Define the volume
volume = 3200**3  # Mpc^3

# Loop over snapshots
for snap in SNAPSHOTS:
    print(f"Plotting {snap}")

    # Get the redshift from the snap tag
    z = float(snap.split("_")[-1].replace("z", "").replace("p", "."))

    # Read the UV1500 luminosities, weights and LRD mask
    with h5py.File(data_file.replace("<snap>", snap), "r") as hdf:
        lnu = hdf[f"Galaxies/Photometry/Luminosities/{args.spec_type}/UV1500"][...]
        weights = hdf["Galaxies/RegionWeight"][...]
        mask = hdf[f"Galaxieis/LRDFlag/{args.spec_type}"][...]

    # Convert flux to absolute magnitude
    mags = lnu_to_absolute_mag(lnu)

    # Compute the luminosity function full
    hist, _ = np.histogram(mags, bins=bins)
    whist, _ = np.histogram(mags, bins=bins, weights=weights)
    hist = hist.astype(float)
    whist = whist.astype(float)
    hist *= whist
    phi = hist / volume / np.diff(bins)

    # Compute the LRD luminosity function (masked LF)
    hist, _ = np.histogram(mags[mask], bins=bins)
    whist, _ = np.histogram(mags[mask], bins=bins, weights=weights[mask])
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
        f"UVLF/luminosity_function_{args.spec_type}_{snap}",
    )


# Define magnitude bins
bins = np.logspace(27, 31, 20)
bin_cents = (bins[:-1] + bins[1:]) / 2

# Define the volume
volume = 3200**3  # Mpc^3

# Loop over snapshots
for snap in SNAPSHOTS:
    # Get the redshift from the snap tag
    z = float(snap.split("_")[-1].replace("z", "").replace("p", "."))

    # Read the UV1500 luminosities, weights and LRD mask
    with h5py.File(data_file.replace("<snap>", snap), "r") as hdf:
        lnu = hdf[f"Galaxies/Photometry/Luminosities/{args.spec_type}/UV1500"][...]
        weights = hdf["Galaxies/RegionWeight"][...]
        mask = hdf[f"Galaxies/LRDFlag/{args.spec_type}"][...]

    # Compute the luminosity function full
    hist, _ = np.histogram(lnu, bins=bins)
    whist, _ = np.histogram(lnu, bins=bins, weights=weights)
    hist = hist.astype(float)
    whist = whist.astype(float)
    hist *= whist
    phi = hist / volume / np.diff(bins)

    # Compute the LRD luminosity function (masked LF)
    hist, _ = np.histogram(lnu[mask], bins=bins)
    whist, _ = np.histogram(lnu[mask], bins=bins, weights=weights[mask])
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
    ax.set_xlabel("$L_{2000} / [erg / s / Hz]$")

    ax.set_ylabel(r"$\phi$ / [Mpc$^{-3}$ dex$^{-1}$]")

    ax.legend()

    savefig(
        fig,
        f"UVLF/luminosity_function_nomag_{args.spec_type}_{snap}",
    )
