"""A script for counting LRDs in different selections."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.cosmology import Planck15 as cosmo

from utils import (
    get_synth_data_with_imgs,
    savefig,
    get_master_data,
    FLUX_LIMIT,
)

from synthesizer.conversions import absolute_mag_to_lnu, lnu_to_fnu

# Define the data file
data_file = "data/combined_<region>_<snap>.hdf5"
master_file = (
    "/cosma7/data/dp004/dc-payy1/my_files//flares_pipeline/data/flares.hdf5"
)

# Get the synthesizer data for combined, stellar and agn
(
    fluxes,
    colors,
    red1,
    red2,
    sizes,
    masks,
    indices,
    images,
) = get_synth_data_with_imgs(data_file, "attenuated")
(
    stellar_fluxes,
    stellar_colors,
    stellar_red1,
    stellar_red2,
    stellar_sizes,
    stellar_masks,
    stellar_indices,
    stellar_images,
) = get_synth_data_with_imgs(data_file, "stellar_attenuated")
(
    agn_fluxes,
    agn_colors,
    agn_red1,
    agn_red2,
    agn_sizes,
    agn_masks,
    agn_indices,
    agn_images,
) = get_synth_data_with_imgs(data_file, "agn_attenuated")


# Get the masses from the master file
masses = get_master_data(master_file, indices, "Mstar_aperture/30")
stellar_masses = get_master_data(
    master_file, stellar_indices, "Mstar_aperture/30"
)
agn_masses = get_master_data(master_file, agn_indices, "Mstar_aperture/30")

# Convert masses to Msun
for snap in masses:
    masses[snap] *= 10**10
    stellar_masses[snap] *= 10**10
    agn_masses[snap] *= 10**10

# Loop over snapshots
for snap in images:
    # Get fluxes
    flux = fluxes[snap]["F444W"][masks[snap]]
    stellar_flux = stellar_fluxes[snap]["F444W"][stellar_masks[snap]]
    agn_flux = agn_fluxes[snap]["F444W"][agn_masks[snap]]

    # Get the masses
    mass = masses[snap][masks[snap]]
    stellar_mass = stellar_masses[snap][stellar_masks[snap]]
    agn_mass = agn_masses[snap][agn_masks[snap]]

    # Get redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Skip empty snapshots
    if len(flux) == 0 or len(stellar_flux) == 0 or len(agn_flux) == 0:
        continue

    # Convert the flux limit to nJy
    flux_limit = lnu_to_fnu(absolute_mag_to_lnu(FLUX_LIMIT), cosmo, z).to(
        "nJy"
    )

    print("Flux limit:", flux_limit)

    # Define flux bins
    flux_bins = np.logspace(
        np.log10(flux_limit),
        np.log10(np.max(flux)),
        30,
    )

    # Define mass bins
    mass_bins = np.logspace(
        np.log10(np.min(mass)),
        np.log10(np.max(mass)),
        30,
    )

    # Count the LRDs
    lrd_counts = np.zeros((len(flux_bins) - 1, len(mass_bins) - 1))
    stellar_lrd_counts = np.zeros((len(flux_bins) - 1, len(mass_bins) - 1))
    agn_lrd_counts = np.zeros((len(flux_bins) - 1, len(mass_bins) - 1))

    for i in range(len(flux_bins) - 1):
        for j in range(len(mass_bins) - 1):
            mask = (
                (flux > flux_bins[i])
                & (flux < flux_bins[i + 1])
                & (mass > mass_bins[j])
                & (mass < mass_bins[j + 1])
            )
            stellar_mask = (
                (stellar_flux > flux_bins[i])
                & (stellar_flux < flux_bins[i + 1])
                & (stellar_mass > mass_bins[j])
                & (stellar_mass < mass_bins[j + 1])
            )
            agn_mask = (
                (agn_flux > flux_bins[i])
                & (agn_flux < flux_bins[i + 1])
                & (agn_mass > mass_bins[j])
                & (agn_mass < mass_bins[j + 1])
            )

            lrd_counts[i, j] = np.sum(mask)
            stellar_lrd_counts[i, j] = np.sum(stellar_mask)
            agn_lrd_counts[i, j] = np.sum(agn_mask)

    # Create a global norm
    norm = mcolors.LogNorm(
        vmin=1, vmax=np.max((lrd_counts, stellar_lrd_counts, agn_lrd_counts))
    )

    # Define the extent
    extent = [
        np.log10(flux_bins[0]),
        np.log10(flux_bins[-1]),
        np.log10(mass_bins[0]),
        np.log10(mass_bins[-1]),
    ]

    # Setup the figure
    fig = plt.figure(figsize=(3 * 3.5 + 0.1, 3.5))

    # Set up the gridspec
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.1], wspace=0.0)

    # Create the axes
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    cax = fig.add_subplot(gs[3])

    # Plot the LRD counts
    im1 = ax1.imshow(
        lrd_counts.T,
        norm=norm,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        extent=extent,
    )
    im2 = ax2.imshow(
        stellar_lrd_counts.T,
        norm=norm,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        extent=extent,
    )
    im3 = ax3.imshow(
        agn_lrd_counts.T,
        norm=norm,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        extent=extent,
    )

    # Add titles
    ax1.set_title("Combined")
    ax2.set_title("Stellar")
    ax3.set_title("AGN")

    # Remove y-ticks where necessary
    ax2.set_yticks([])
    ax3.set_yticks([])

    # Add the colorbar
    cbar = fig.colorbar(im3, cax=cax)
    cbar.set_label(r"$N_\mathrm{LRD}$")

    savefig(fig, f"LRD_counts/lrd_counts_2D_{snap}.pdf")
