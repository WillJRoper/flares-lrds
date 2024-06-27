"""A module containing the definition of helpful functions."""
import h5py
import numpy as np
import matplotlib.pyplot as plt

from unyt import unyt_array


# Get regions and snapshots
REGIONS = [str(reg).zfill(2) for reg in range(40)]
SNAPSHOTS = [
    "005_z010p000",
    "006_z009p000",
    "007_z008p000",
    "008_z007p000",
    "009_z006p000",
    "010_z005p000",
]

# Define a dictionary with the Kokorev+24 thresholds
KOKOREV24 = {
    "F115W_F150W": 0.8,
    "F150W_F200W": 0.8,
    "F200W_F277W": 0.7,
    "F200W_F356W": 1.0,
    "F277W_F356W": 0.6,
    "F277W_F444W": 0.7,
}

# Define the list of filter IDs
FILTER_CODES = [
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F444W",
]


def get_sizes_mdot(master_file_path):
    """
    Get the sizes and accretion rates of the black holes in the simulation.

    Args:
        master_file_path (str): The path to the master file.
    """
    # Define containers for the data
    sizes = {}
    mdots = {}

    # Open the master file
    with h5py.File(master_file_path, "r") as hdf:
        # Loop over regions
        for reg in REGIONS:
            # Loop over snapshots
            for snap in SNAPSHOTS:
                # Ensure a key exists for this snapshot
                sizes.setdefault(snap, {})
                mdots.setdefault(snap, {})

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
                        np.max(
                            hdf[f"{reg}/{snap}/Particle/BH_Mdot"][start:end]
                        )
                        * 10**10
                        / 0.6777
                    )

                # Store the data
                sizes[snap].setdefault("size", []).extend(size)
                mdots[snap].setdefault("mdot", []).extend(mdot)

    # Convert the data to arrays
    for snap in sizes.keys():
        sizes[snap]["size"] = np.array(sizes[snap]["size"])
        mdots[snap]["mdot"] = np.array(mdots[snap]["mdot"])

    return sizes, mdots


def get_synth_data(synth_data_path, spec, size_thresh=1):
    """
    Get the fluxes and colors of the galaxies in the simulation.

    Args:
        synth_data_path (str): The path to the synthetic data.
        spec (str): The spectral synthesis model to use.
        size_thresh (float): The size threshold to apply for LRDs.

    Returns:
        dict: A dictionary containing the fluxes of the galaxies.
        dict: A dictionary containing the colors of the galaxies.
        dict: A dictionary containing the red1 mask of the galaxies.
        dict: A dictionary containing the red2 mask of the galaxies.
    """
    # Define containers for the data
    fluxes = {}
    sizes = {}
    indices = {}

    # Loop over regions
    for reg in REGIONS:
        # Loop over snapshots
        for snap in SNAPSHOTS:
            # Ensure a key exists for this snapshot
            fluxes.setdefault(snap, {})
            sizes.setdefault(snap, {})
            indices.setdefault(snap, {})

            # Get the fluxes we need
            with h5py.File(
                synth_data_path.replace("<region>", reg).replace(
                    "<snap>", snap
                ),
                "r",
            ) as hdf:
                try:
                    inds = hdf["Indices"][...]
                    f115w = unyt_array(
                        hdf[f"ObservedPhotometry/{spec}/JWST/NIRCam.F115W"][
                            ...
                        ],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f150w = unyt_array(
                        hdf[f"ObservedPhotometry/{spec}/JWST/NIRCam.F150W"][
                            ...
                        ],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f200w = unyt_array(
                        hdf[f"ObservedPhotometry/{spec}/JWST/NIRCam.F200W"][
                            ...
                        ],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f277w = unyt_array(
                        hdf[f"ObservedPhotometry/{spec}/JWST/NIRCam.F277W"][
                            ...
                        ],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f356w = unyt_array(
                        hdf[f"ObservedPhotometry/{spec}/JWST/NIRCam.F356W"][
                            ...
                        ],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f444w = unyt_array(
                        hdf[f"ObservedPhotometry/{spec}/JWST/NIRCam.F444W"][
                            ...
                        ],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    for filt in FILTER_CODES:
                        sizes[snap].setdefault(filt.split(".")[-1], []).extend(
                            hdf[f"HalfLightRadii/{spec}/{filt}"][...]
                        )

                except KeyError as e:
                    print(f"KeyError: {e}")
                    continue
                except OSError as e:
                    print(f"OSError: {e}")
                    continue
                except TypeError as e:
                    print(f"TypeError: {e}")
                    continue

                # Store the data
                fluxes[snap].setdefault("F115W", []).extend(f115w)
                fluxes[snap].setdefault("F150W", []).extend(f150w)
                fluxes[snap].setdefault("F200W", []).extend(f200w)
                fluxes[snap].setdefault("F277W", []).extend(f277w)
                fluxes[snap].setdefault("F356W", []).extend(f356w)
                fluxes[snap].setdefault("F444W", []).extend(f444w)
                indices[snap].setdefault(reg, []).extend(inds)

    # Convert the data to arrays
    for snap in fluxes.keys():
        for key in fluxes[snap].keys():
            fluxes[snap][key] = np.array(fluxes[snap][key])
            sizes[snap][key] = np.array(sizes[snap][key])
            for reg in REGIONS:
                if reg in indices[snap]:
                    indices[snap][reg] = np.array(
                        indices[snap][reg], dtype=int
                    )
                else:
                    indices[snap][reg] = np.array([], dtype=int)

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

    # Derive the kokorev masks
    red1 = {}
    red2 = {}
    for snap in colors.keys():
        mask = np.logical_and(
            colors[snap]["F115W_F150W"] < 0.8,
            colors[snap]["F200W_F277W"] > 0.7,
        )
        red1[snap] = np.logical_and(mask, colors[snap]["F200W_F356W"] > 1.0)
        mask = np.logical_and(
            colors[snap]["F150W_F200W"] < 0.8,
            colors[snap]["F277W_F356W"] > 0.6,
        )
        red2[snap] = np.logical_and(mask, colors[snap]["F277W_F444W"] > 0.7)

    # Combine the masks with a size threshold
    masks = {}
    for snap in sizes.keys():
        mask = np.logical_and(
            np.logical_or(red1[snap], red2[snap]),
            sizes[snap]["F444W"] < size_thresh,
        )
        masks[snap] = mask

    return fluxes, colors, red1, red2, sizes, masks, indices


def get_master_data(master_file_path, indices, key):
    """
    Get the data from the master file for the given indices.

    Args:
        master_file_path (str): The path to the master file.
        indices (list): The indices of the galaxies to extract.
        key (str): The key of the data to extract.

    Returns:
        dict: A dictionary containing the data for the galaxies.
    """
    # Open the file
    with h5py.File(master_file_path, "r") as hdf:
        # Define the data dictionary
        data = {}

        # Loop over regions
        for reg in REGIONS:
            # Loop over snapshots
            for snap in SNAPSHOTS:
                # Ensure a key exists for this snapshot
                data.setdefault(snap, [])

                # Get the data we need
                data[snap].extend(
                    hdf[f"{reg}/{snap}/Galaxy/{key}"][indices[snap][reg]]
                )

    return data


def get_masked_synth_data(synth_path, key, masks):
    """
    Get synthesizer galaxy data including a mask.

    Args:
        synth_path (str): The path to the synthetic data.
        key (str): The key of the data to extract.
        masks (dict): The masks to apply.
    """
    # Define containers for the data
    data = {}

    # Loop over regions
    for reg in REGIONS:
        # Loop over snapshots
        for snap in SNAPSHOTS:
            # Ensure a key exists for this snapshot
            data.setdefault(snap, [])

            # Get the data we need
            with h5py.File(
                synth_path.replace("<region>", reg).replace("<snap>", snap),
                "r",
            ) as hdf:
                try:
                    data[snap].extend(hdf[key][...])
                    print(f"reading data {reg} {snap}")
                except KeyError as e:
                    print(f"KeyError: {e}")
                    continue
                except OSError as e:
                    print(f"OSError: {e}")
                    continue
                except TypeError as e:
                    print(f"TypeError: {e}")
                    continue

    # Apply the mask
    for snap in data.keys():
        data[snap] = np.array(data[snap])[masks[snap]]

    return data


def plot_masked_unmasked_hexbins(
    xs,
    ys,
    mask,
    extent,
    norm,
    xlabel,
    ylabel,
    basename,
    gridsize=50,
    cmap="viridis",
    xscale="log",
    yscale="log",
):
    """
    Plot the masked and unmasked hexbins.

    Args:
        xs (array): The x-values to plot.
        ys (array): The y-values to plot.
        mask (array): The mask to apply.
        extent (tuple): The extent of the plot.
        norm (Normalize): The normalization to use.
        xlabel (str): The x-axis label.
        ylabel (str): The y-axis label.
        basename (str): The base name of the plot.
        gridsize (int): The number of bins to use.
        cmap (str): The colormap to use.
        xscale (str): The x-axis scale.
        yscale (str): The y-axis scale.
    """
    # Plot the size-luminosity relation
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))

    # Draw the grid and make sure its in the background
    axs[0].set_axisbelow(True)
    axs[1].set_axisbelow(True)
    axs[0].grid(True)
    axs[1].grid(True)

    # Plot the size-luminosity relation with no mask
    axs[0].hexbin(
        xs,
        ys,
        gridsize=gridsize,
        norm=norm,
        extent=extent,
        cmap=cmap,
        linewidths=0.2,
        xscale=xscale,
        yscale=yscale,
        mincnt=1,
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

    # Plot the size-luminosity relation with the mask
    axs[1].hexbin(
        xs[mask],
        ys[mask],
        gridsize=gridsize,
        norm=norm,
        extent=extent,
        cmap=cmap,
        linewidths=0.2,
        xscale=xscale,
        yscale=yscale,
        mincnt=1,
    )
    axs[1].text(
        0.95,
        0.05,
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

    # Set the labels
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[1].set_xlabel(xlabel)

    # Remove the y-axis label from the right plot
    axs[1].set_yticklabels([])

    # Draw the colorbar
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axs,
        orientation="vertical",
        pad=0.05,
        aspect=30,
    )
    cb.set_label("$N$")

    # Save the figure
    fig.savefig(
        f"plots/{basename}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
