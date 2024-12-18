"""A script for plotting the luminsoity function of different galaxy types."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from synthesizer.conversions import lnu_to_absolute_mag
from unyt import unyt_array

from utils import SNAPSHOTS, SPECTRA_KEYS, savefig


def read_gal_data(snap, component, spec_type):
    """Read the luminosity data from the HDF5 file."""
    # Get the redshift from the snap tag
    z = float(snap.split("_")[-1].replace("z", "").replace("p", "."))

    # Read the UV1500 luminosities, weights and LRD mask
    try:
        with h5py.File(data_file.replace("<snap>", snap), "r") as hdf:
            lnu_dset = hdf[
                f"Galaxies/{component}Photometry/Luminosities/{spec_type}/UV1500"
            ]
            lnu = unyt_array(lnu_dset[...], lnu_dset.attrs["Units"])
            weights = hdf["Galaxies/RegionWeight"][...]
            mask = hdf[f"Galaxies/{component}LRDFlag/{spec_type}"][...]
    except OSError as e:
        print(e)
        return None, None, None, None

    return lnu, weights, mask, z


def plot_lf(
    data,
    weights,
    mask,
    bins,
    volume,
    bin_cents,
    bin_widths,
    outpath,
    xlabel,
    ylabel,
    xscale="linear",
):
    """Plot the luminosity function."""
    # Compute the luminosity function full
    whist, _ = np.histogram(data, bins=bins, weights=weights)
    phi = whist / volume / bin_widths

    # Compute the LRD luminosity function (masked LF)
    whist, _ = np.histogram(data[mask], bins=bins, weights=weights[mask])
    lrd_phi = whist / volume / bin_widths

    # Plot the luminosity function
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xscale(xscale)

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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    savefig(fig, outpath)


# Define the data file
data_file = "data/combined_<snap>.hdf5"

# Define magnitude and luminosity bins
mag_bins = np.arange(-27, -16, 1)
mag_bin_cents = (mag_bins[:-1] + mag_bins[1:]) / 2
mag_bin_widths = np.diff(mag_bins)
lum_bins = np.logspace(27, 31, 20)
lum_bin_cents = (lum_bins[:-1] + lum_bins[1:]) / 2
lum_bin_widths = np.diff(np.log10(lum_bins))

# Define the volume
volume = vol = (4 / 3) * np.pi * (14 / 0.6777) ** 3  # Mpc^3

for spec_type, component in zip(
    SPECTRA_KEYS,
    ["Stars/", "BlackHoles/", "Stars/", "BlackHoles/", "Stars/", "", "", ""],
):
    for snap in SNAPSHOTS:
        print(f"Plotting {snap}")

        # Get the data for galaxies
        lnu, weights, mask, z = read_gal_data(snap, component, spec_type)

        print(f"{spec_type}: Number of galaxies: {len(lnu)}")
        print(f"{spec_type}: Number of LRDs: {len(lnu[mask])}")

        # Convert flux to absolute magnitude
        mags = lnu_to_absolute_mag(lnu)

        # PLot the luminosity function
        plot_lf(
            mags,
            weights,
            mask,
            mag_bins,
            volume,
            mag_bin_cents,
            mag_bin_widths,
            f"UVLF/mag_luminosity_function_{spec_type}_{snap}",
            "$M_{1500}$",
            r"$\phi$ / [Mpc$^{-3}$ mag$^{-1}$]",
        )
        plot_lf(
            lnu,
            weights,
            mask,
            lum_bins,
            volume,
            lum_bin_cents,
            lum_bin_widths,
            f"UVLF/luminosity_function_{spec_type}_{snap}",
            "$L_{1500} / [erg / s / Hz]$",
            r"$\phi$ / [Mpc$^{-3}$ dex$^{-1}$]",
            xscale="log",
        )
