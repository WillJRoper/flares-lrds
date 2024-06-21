"""A script for plotting the color distribution of LRDs in FLARES."""

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.cosmology import Planck15 as cosmo

# Define the master file path
master_file_path = sys.argv[1]

# Define containers for the data
fluxes = {}
sizes = {}
mdots = {}

# Open the master file
with h5py.File(master_file_path, "r") as hdf:
    # Loop over regions
    for region in range(0, 40):
        reg = str(region).zfill(2)

        # Loop over snapshots
        for snap in hdf[reg].keys():
            # Ensure a key exists for this snapshot
            fluxes.setdefault(snap, {})
            sizes.setdefault(snap, {})
            mdots.setdefault(snap, {})

            # Get the fluxes we need
            f115w = hdf[
                f"{reg}/{snap}/Galaxy/BPASS_2.2.1/Chabrier300"
                "/Flux/DustModelI/JWST/NIRCAM/F115W"
            ][...]
            f150w = hdf[
                f"{reg}/{snap}/Galaxy/BPASS_2.2.1/Chabrier300"
                "/Flux/DustModelI/JWST/NIRCAM/F150W"
            ][...]
            f200w = hdf[
                f"{reg}/{snap}/Galaxy/BPASS_2.2.1/Chabrier300"
                "/Flux/DustModelI/JWST/NIRCAM/F200W"
            ][...]
            f277w = hdf[
                f"{reg}/{snap}/Galaxy/BPASS_2.2.1/Chabrier300"
                "/Flux/DustModelI/JWST/NIRCAM/F277W"
            ][...]
            f356w = hdf[
                f"{reg}/{snap}/Galaxy/BPASS_2.2.1/Chabrier300"
                "/Flux/DustModelI/JWST/NIRCAM/F356W"
            ][...]
            f444w = hdf[
                f"{reg}/{snap}/Galaxy/BPASS_2.2.1/Chabrier300"
                "/Flux/DustModelI/JWST/NIRCAM/F444W"
            ][...]

            # Get the sizes we need
            size = hdf[f"{reg}/{snap}/Galaxy/HalfMassRad"][:, 4] * 1000

            # Get the black holes slices for each galaxy
            bh_len = hdf[f"{reg}/{snap}/Galaxy/BH_Length"]
            bh_start = np.concatenate(([0], np.cumsum(bh_len)))[:-1]
            bh_end = bh_start + bh_len

            # Get the accretion rates
            mdot = np.zeros_like(size)
            for i, (start, end) in enumerate(zip(bh_start, bh_end)):
                if end - start == 0:
                    continue
                mdot[i] = (
                    np.max(hdf[f"{reg}/{snap}/Particle/BH_Mdot"][start:end])
                    * 10**10
                    / 0.6777
                )

            # Store the data
            fluxes[snap].setdefault("F115W", []).extend(f115w)
            fluxes[snap].setdefault("F150W", []).extend(f150w)
            fluxes[snap].setdefault("F200W", []).extend(f200w)
            fluxes[snap].setdefault("F277W", []).extend(f277w)
            fluxes[snap].setdefault("F356W", []).extend(f356w)
            fluxes[snap].setdefault("F444W", []).extend(f444w)
            sizes[snap].setdefault("size", []).extend(size)
            mdots[snap].setdefault("mdot", []).extend(mdot)

# Convert the data to arrays
for snap in fluxes.keys():
    for key in fluxes[snap].keys():
        fluxes[snap][key] = np.array(fluxes[snap][key])
    sizes[snap]["size"] = np.array(sizes[snap]["size"])

# Compute the colors
colors = {}
for snap in fluxes.keys():
    colors.setdefault(snap, {})
    colors[snap]["F115W_F150W"] = -2.5 * np.log10(
        fluxes[snap]["F115W"] / fluxes[snap]["F150W"]
    )
    colors[snap]["F150W_F200W"] = -2.5 * np.log10(
        fluxes[snap]["F150W"] / fluxes[snap]["F200W"]
    )
    colors[snap]["F200W_F277W"] = -2.5 * np.log10(
        fluxes[snap]["F200W"] / fluxes[snap]["F277W"]
    )
    colors[snap]["F200W_F356W"] = -2.5 * np.log10(
        fluxes[snap]["F200W"] / fluxes[snap]["F356W"]
    )
    colors[snap]["F200W_F277W"] = -2.5 * np.log10(
        fluxes[snap]["F200W"] / fluxes[snap]["F277W"]
    )
    colors[snap]["F277W_F356W"] = -2.5 * np.log10(
        fluxes[snap]["F277W"] / fluxes[snap]["F356W"]
    )
    colors[snap]["F277W_F444W"] = -2.5 * np.log10(
        fluxes[snap]["F277W"] / fluxes[snap]["F444W"]
    )

# Define plot props
gridsize = 50
norm = LogNorm(vmin=1, vmax=10**4.1)

# Define a dictionary with the Kokorev+24 thresholds
kokorev24 = {
    "F115W_F150W": 0.8,
    "F150W_F200W": 0.8,
    "F200W_F277W": 0.7,
    "F200W_F356W": 1.0,
    "F277W_F356W": 0.6,
    "F277W_F444W": 0.7,
}

# Derive the kokorev masks
red1 = {}
red2 = {}
for snap in colors.keys():
    mask = np.logical_and(
        colors[snap]["F115W_F150W"] < 0.8, colors[snap]["F200W_F277W"] > 0.7
    )
    red1[snap] = np.logical_and(mask, colors[snap]["F200W_F356W"] > 1.0)
    mask = np.logical_and(
        colors[snap]["F150W_F200W"] < 0.8, colors[snap]["F277W_F356W"] > 0.6
    )
    red2[snap] = np.logical_and(mask, colors[snap]["F277W_F444W"] > 0.7)

# Make the color hexbin plots for each snapshot
for snap in colors.keys():
    # Create the figure
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Plot the color-color distributions
    ax[0, 0].hexbin(
        colors[snap]["F200W_F277W"],
        colors[snap]["F200W_F356W"],
        gridsize=gridsize,
        cmap="viridis",
        linewidth=0.2,
        mincnt=1,
        norm=norm,
        extent=[-1, 2.5, -1, 4],
    )
    ax[0, 0].scatter(
        colors[snap]["F200W_F277W"][red1[snap]],
        colors[snap]["F200W_F356W"][red1[snap]],
        color="red",
        s=3,
        alpha=0.7,
    )
    ax[0, 0].set_xlabel("F200W - F277W")
    ax[0, 0].set_ylabel("F200W - F356W")
    ax[0, 0].text(
        0.05,
        0.95,
        "Red 1 (Kokorev+24)",
        transform=ax[0, 0].transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    ax[0, 0].set_xlim(-1, 2.5)
    ax[0, 0].set_ylim(-1, 4)

    ax[0, 1].hexbin(
        colors[snap]["F277W_F356W"],
        colors[snap]["F277W_F444W"],
        gridsize=gridsize,
        cmap="viridis",
        linewidth=0.2,
        mincnt=1,
        norm=norm,
        extent=[-2, 2.5, -1, 4],
    )
    ax[0, 1].scatter(
        colors[snap]["F277W_F356W"][red2[snap]],
        colors[snap]["F277W_F444W"][red2[snap]],
        color="red",
        s=3,
        alpha=0.7,
    )
    ax[0, 1].set_xlabel("F277W - F356W")
    ax[0, 1].set_ylabel("F277W - F444W")
    ax[0, 1].text(
        0.05,
        0.95,
        "Red 2 (Kokorev+24)",
        transform=ax[0, 1].transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    ax[0, 1].set_xlim(-2, 2.5)
    ax[0, 1].set_ylim(-1, 4)

    ax[1, 0].hexbin(
        colors[snap]["F200W_F277W"],
        colors[snap]["F115W_F150W"],
        gridsize=gridsize,
        cmap="viridis",
        linewidth=0.2,
        mincnt=1,
        norm=norm,
        extent=[-1, 2.5, -1, 4],
    )
    ax[1, 0].scatter(
        colors[snap]["F200W_F277W"][red1[snap]],
        colors[snap]["F115W_F150W"][red1[snap]],
        color="red",
        s=3,
        alpha=0.7,
    )
    ax[1, 0].text(
        0.05,
        0.95,
        "Red 1 (Kokorev+24)",
        transform=ax[1, 0].transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    ax[1, 0].set_xlabel("F200W - F277W")
    ax[1, 0].set_ylabel("F115W - F150W")
    ax[1, 0].set_xlim(-1, 2.5)
    ax[1, 0].set_ylim(-1, 4)

    ax[1, 1].hexbin(
        colors[snap]["F277W_F356W"],
        colors[snap]["F115W_F150W"],
        gridsize=gridsize,
        cmap="viridis",
        linewidth=0.2,
        mincnt=1,
        norm=norm,
        extent=[-2, 2.5, -1, 4],
    )
    ax[1, 1].scatter(
        colors[snap]["F277W_F356W"][red2[snap]],
        colors[snap]["F115W_F150W"][red2[snap]],
        color="red",
        s=3,
        alpha=0.7,
    )
    ax[1, 1].set_xlabel("F277W - F356W")
    ax[1, 1].set_ylabel("F115W - F150W")
    ax[1, 1].text(
        0.05,
        0.95,
        "Red 2 (Kokorev+24)",
        transform=ax[1, 1].transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    ax[1, 1].set_xlim(-2, 2.5)
    ax[1, 1].set_ylim(-1, 4)

    # Turn on the grid for each axis
    ax[0, 0].grid(True)
    ax[0, 1].grid(True)
    ax[1, 0].grid(True)
    ax[1, 1].grid(True)

    # Add a colorbar
    cbar = fig.colorbar(ax[0, 0].collections[0], ax=ax)
    cbar.set_label("Number of galaxies")

    # Save the figure
    fig.savefig(f"lrd_kokorev_colors_{snap}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Plot the size mass relation flagging "red" galaxies
for snap in sizes.keys():
    # Get redshift
    z = float(snap.split("_")[-1].replace("z", "").replace("p", "."))

    # Create the figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Logscale both plots
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xscale("log")

    # Get a mask
    okinds = np.logical_and(fluxes[snap]["F444W"] > 0, sizes[snap]["size"] > 0)

    # Plot the size-mass relation
    ax[0].hexbin(
        fluxes[snap]["F444W"][okinds],
        sizes[snap]["size"][okinds],
        gridsize=gridsize,
        cmap="viridis",
        linewidth=0.2,
        mincnt=1,
        norm=norm,
        xscale="log",
        yscale="log",
    )
    ax[0].scatter(
        fluxes[snap]["F444W"][np.logical_and(red1[snap], okinds)],
        sizes[snap]["size"][np.logical_and(red1[snap], okinds)],
        color="red",
        s=3,
        alpha=0.7,
    )
    ax[0].text(
        0.05,
        0.95,
        "Red 1 (Kokorev+24)",
        transform=ax[0].transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    ax[0].set_ylabel("Half mass radius (pkpc)")
    ax[0].set_xlabel("F444W (nJy)")

    ax[1].hexbin(
        fluxes[snap]["F444W"][okinds],
        sizes[snap]["size"][okinds],
        gridsize=gridsize,
        cmap="viridis",
        linewidth=0.2,
        mincnt=1,
        norm=norm,
        xscale="log",
        yscale="log",
    )
    ax[1].scatter(
        fluxes[snap]["F444W"][np.logical_and(red2[snap], okinds)],
        sizes[snap]["size"][np.logical_and(red2[snap], okinds)],
        color="red",
        s=3,
        alpha=0.7,
    )
    ax[1].text(
        0.05,
        0.95,
        "Red 2 (Kokorev+24)",
        transform=ax[1].transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    ax[1].set_xlabel("F444W (nJy)")

    # Turn on the grid for each axis
    ax[0].grid(True)
    ax[1].grid(True)

    # Add a colorbar
    cbar = fig.colorbar(ax[0].collections[0], ax=ax, pad=0.1)
    cbar.set_label("Number of galaxies")

    # Add an alternative y axis to the right hand plot in arcseconds
    ax2 = ax[1].secondary_yaxis(
        "right",
        functions=(
            lambda x: x * cosmo.arcsec_per_kpc_proper(z).value,
            lambda x: x / cosmo.arcsec_per_kpc_proper(z).value,
        ),
    )
    ax2.set_ylabel("Half mass radius (arcsecond)")

    # Save the figure
    fig.savefig(f"lrd_kokorev_sizes_{snap}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Plot size vs color coloured by mdot
for snap in mdots.keys():
    # Convert mdots to an array
    mdots[snap]["mdot"] = np.array(mdots[snap]["mdot"])

    # Get the normalisation
    norm = LogNorm(
        vmin=np.min(mdots[snap]["mdot"][mdots[snap]["mdot"] > 0]) * 10,
        vmax=np.max(mdots[snap]["mdot"]),
    )

    for i, color in enumerate(colors[snap].keys()):
        # Create the figure
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        # Logscale both plots
        ax[0, 0].set_yscale("log")
        ax[1, 0].set_yscale("log")
        ax[1, 1].set_yscale("log")

        # Include a grid
        ax[0, 0].grid(True)
        ax[1, 0].grid(True)
        ax[1, 1].grid(True)

        # Remove the axis in the top right corner
        ax[0, 1].axis("off")

        # Plot the size-color relation
        sc = ax[0, 0].hexbin(
            colors[snap][color],
            sizes[snap]["size"],
            C=mdots[snap]["mdot"],
            gridsize=gridsize,
            cmap="viridis",
            reduce_C_function=np.mean,
            linewidth=0.2,
            mincnt=np.min(mdots[snap]["mdot"][mdots[snap]["mdot"] > 0]),
            yscale="log",
            norm=norm,
        )
        ax[1, 0].hexbin(
            colors[snap][color][red1[snap]],
            sizes[snap]["size"][red1[snap]],
            C=mdots[snap]["mdot"][red1[snap]],
            cmap="viridis",
            gridsize=gridsize,
            reduce_C_function=np.mean,
            linewidth=0.2,
            mincnt=np.min(mdots[snap]["mdot"][mdots[snap]["mdot"] > 0]),
            yscale="log",
            norm=norm,
        )
        ax[1, 1].hexbin(
            colors[snap][color][red2[snap]],
            sizes[snap]["size"][red2[snap]],
            C=mdots[snap]["mdot"][red2[snap]],
            cmap="viridis",
            gridsize=gridsize,
            reduce_C_function=np.mean,
            linewidth=0.2,
            mincnt=np.min(mdots[snap]["mdot"][mdots[snap]["mdot"] > 0]),
            yscale="log",
            norm=norm,
        )

        # Label the axes
        ax[0, 0].set_xlabel(color)
        ax[0, 0].set_ylabel("Half mass radius (pkpc)")
        ax[1, 0].set_xlabel(color)
        ax[1, 0].set_ylabel("Half mass radius (pkpc)")
        ax[1, 1].set_xlabel(color)

        # Add text label
        ax[1, 0].text(
            0.05,
            0.95,
            "Red 1 (Kokorev+24)",
            transform=ax[1, 0].transAxes,
            fontsize=12,
            verticalalignment="top",
        )
        ax[1, 1].text(
            0.05,
            0.95,
            "Red 2 (Kokorev+24)",
            transform=ax[1, 1].transAxes,
            fontsize=12,
            verticalalignment="top",
        )

        # Add an alternative y axis to the right hand plot in arcseconds
        ax2 = ax[0, 0].secondary_yaxis(
            "right",
            functions=(
                lambda x: x * cosmo.arcsec_per_kpc_proper(z).value,
                lambda x: x / cosmo.arcsec_per_kpc_proper(z).value,
            ),
        )
        ax3 = ax[1, 1].secondary_yaxis(
            "right",
            functions=(
                lambda x: x * cosmo.arcsec_per_kpc_proper(z).value,
                lambda x: x / cosmo.arcsec_per_kpc_proper(z).value,
            ),
        )
        ax2.set_ylabel("Half mass radius (arcsecond)")
        ax3.set_ylabel("Half mass radius (arcsecond)")

        # Add a colorbar
        cbar = fig.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label("Mean BH accretion rate (Msun/yr)")

        # Save the figure
        fig.savefig(
            f"lrd_kokorev_{color}_mdot_{snap}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
