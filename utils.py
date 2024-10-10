"""A module containing the definition of helpful functions."""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI as mpi
from synthesizer.conversions import apparent_mag_to_fnu
from unyt import kpc, nJy, unyt_array

# Define the flux limit in F444W (this is an apparent AB magnitude)
FLUX_LIMIT = 28.3

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
    "UV1500",
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F444W",
]
RUBIES_FILTER_CODES = [
    "UV1500",
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F140M",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F162M",
    "JWST/NIRCam.F182M",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F210M",
    "JWST/NIRCam.F250M",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F300M",
    "JWST/NIRCam.F335M",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F360M",
    "JWST/NIRCam.F410M",
    "JWST/NIRCam.F430M",
    "JWST/NIRCam.F444W",
    "JWST/NIRCam.F460M",
    "JWST/NIRCam.F480M",
    "JWST/MIRI.F770W",
    "JWST/MIRI.F1800W",
]
TOPHATS = {
    "UV1500": {"lam_eff": 1500, "lam_fwhm": 300},
}

# Define spectra keys
SPECTRA_KEYS = [
    "old_transmitted",
    "young_transmitted",
    "old_nebular",
    "young_nebular",
    "old_reprocessed",
    "young_reprocessed",
    "old_escaped",
    "young_escaped",
    "young_attenuated",
    "old_attenuated",
    "young_intrinsic",
    "old_intrinsic",
    "stellar_intrinsic",
    "agn_intrinsic",
    "escaped",
    "stellar_attenuated",
    "agn_attenuated",
    "stellar_total",
    "combined_intrinsic",
    "total_dust_free_agn",
    "total",
]


def _print(*args, **kwargs):
    """Overload print with rank info."""
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"[{str(rank).zfill(len(str(size)) + 1)}]: ", end="")
    print(*args, **kwargs)


def savefig(fig, outname):
    """
    Save a figure to the plots directory.

    Args:
        fig (Figure): The figure to save.
        outname (str): The name of the output file.
    """
    # Split the directory and file name
    dirname, basename = os.path.split(outname)

    # Create the directory if it doesn't exist
    if dirname:
        os.makedirs("plots/" + dirname, exist_ok=True)

    # Save the figure
    fig.savefig(
        f"plots/{outname}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


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
                        np.max(hdf[f"{reg}/{snap}/Particle/BH_Mdot"][start:end])
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


def get_synth_data(synth_data_path, spec, size_thresh=1, get_weights=False):
    """
    Get the fluxes and colors of the galaxies in the simulation.

    Args:
        synth_data_path (str): The path to the synthetic data.
        spec (str): The spectral synthesis model to use.
        size_thresh (float): The size threshold to apply for LRDs.
        get_weights (bool): Whether to get the weights.

    Returns:
        dict: A dictionary containing the fluxes of the galaxies.
        dict: A dictionary containing the colors of the galaxies.
        dict: A dictionary containing the red1 mask of the galaxies.
        dict: A dictionary containing the red2 mask of the galaxies.
    """
    # Define containers for the data
    fluxes = {}
    sizes = {}
    indices_by_region = {}
    indices = {}
    compactness = {}
    weights = {}

    # If requested, get the weights
    if get_weights:
        region_weights = np.loadtxt("data/weights.txt", usecols=8, delimiter=",")

    # Lood over regions
    for reg in REGIONS:
        # Loop over snapshots
        for snap in SNAPSHOTS:
            # Ensure a key exists for this snapshot
            fluxes.setdefault(snap, {})
            sizes.setdefault(snap, {})
            indices_by_region.setdefault(snap, {})
            compactness.setdefault(snap, [])
            weights.setdefault(snap, [])

            # Get the redshift
            z = float(snap.split("z")[-1].replace("p", "."))

            # Skip files that don't exist
            if not os.path.exists(
                synth_data_path.replace("<region>", reg).replace("<snap>", snap)
            ):
                missing_path = synth_data_path.replace("<region>", reg).replace(
                    "<snap>", snap
                )
                print(f"File {missing_path} does not exist.")
                continue

            # Convert the flux limit to nJy
            flux_limit = apparent_mag_to_fnu(FLUX_LIMIT).to("nJy")

            # Get the fluxes we need
            with h5py.File(
                synth_data_path.replace("<region>", reg).replace("<snap>", snap),
                "r",
            ) as hdf:
                try:
                    # Get f444w fluxes first so we can apply the flux limit,
                    # we always apply this cut to the combined emission
                    f444w = unyt_array(
                        hdf[
                            "ImageObservedPhotometry/"
                            + spec.replace("agn_", "").replace("stellar_", "")
                            + "/JWST/NIRCam.F444W"
                        ][...],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    mask = np.where(f444w > flux_limit)
                    if np.sum(mask) == 0:
                        continue

                    # Apply the mask
                    f444w = f444w[mask]

                    # Get the indices
                    inds = hdf["Indices"][mask]
                    f115w = unyt_array(
                        hdf[f"ImageObservedPhotometry/{spec}/JWST/NIRCam.F115W"][mask],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f150w = unyt_array(
                        hdf[f"ImageObservedPhotometry/{spec}/JWST/NIRCam.F150W"][mask],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f200w = unyt_array(
                        hdf[f"ImageObservedPhotometry/{spec}/JWST/NIRCam.F200W"][mask],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f277w = unyt_array(
                        hdf[f"ImageObservedPhotometry/{spec}/JWST/NIRCam.F277W"][mask],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f356w = unyt_array(
                        hdf[f"ImageObservedPhotometry/{spec}/JWST/NIRCam.F356W"][mask],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f444w = unyt_array(
                        hdf[f"ImageObservedPhotometry/{spec}/JWST/NIRCam.F444W"][mask],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    for filt in FILTER_CODES:
                        sizes[snap].setdefault(filt.split(".")[-1], []).extend(
                            hdf[f"HalfLightRadii/{spec}/{filt}"][mask]
                        )
                    comp = (
                        hdf[f"Apertures/0p4/{spec}/JWST/NIRCam.F444W"][mask]
                        / hdf[f"Apertures/0p2/{spec}/JWST/NIRCam.F444W"][mask]
                    )

                    if get_weights:
                        weights[snap].extend(
                            np.full(len(comp), region_weights[int(reg)])
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
                indices_by_region[snap].setdefault(reg, []).extend(inds)
                compactness[snap].extend(comp)

    # Convert the data to arrays
    for snap in fluxes.keys():
        for key in fluxes[snap].keys():
            fluxes[snap][key] = np.array(fluxes[snap][key]) * nJy
            sizes[snap][key] = np.array(sizes[snap][key]) * kpc
            compactness[snap] = np.array(compactness[snap])
            for reg in REGIONS:
                if reg in indices_by_region[snap]:
                    indices_by_region[snap][reg] = np.array(
                        indices_by_region[snap][reg], dtype=int
                    )
                else:
                    indices_by_region[snap][reg] = np.array([], dtype=int)

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
            compactness[snap] < 1.7,
        )
        masks[snap] = mask

        if get_weights:
            weights[snap] = np.array(weights[snap])

    if get_weights:
        return (
            fluxes,
            colors,
            red1,
            red2,
            sizes,
            masks,
            indices_by_region,
            weights,
        )
    else:
        return fluxes, colors, red1, red2, sizes, masks, indices_by_region


def get_synth_data_with_imgs(synth_data_path, spec):
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
    indices_by_region = {}
    images = {}
    compactness = {}

    # Lood over regions
    for reg in REGIONS:
        # Loop over snapshots
        for snap in SNAPSHOTS:
            # Ensure a key exists for this snapshot
            fluxes.setdefault(snap, {})
            sizes.setdefault(snap, {})
            indices_by_region.setdefault(snap, {})
            images.setdefault(snap, {})
            compactness.setdefault(snap, [])

            # Get the redshift
            z = float(snap.split("z")[-1].replace("p", "."))

            # Skip files that don't exist
            if not os.path.exists(
                synth_data_path.replace("<region>", reg).replace("<snap>", snap)
            ):
                missing_path = synth_data_path.replace("<region>", reg).replace(
                    "<snap>", snap
                )
                print(f"File {missing_path} does not exist.")
                continue

            # Convert the flux limit to nJy
            flux_limit = apparent_mag_to_fnu(FLUX_LIMIT).to("nJy")

            # Get the fluxes we need
            with h5py.File(
                synth_data_path.replace("<region>", reg).replace("<snap>", snap),
                "r",
            ) as hdf:
                try:
                    # Get f444w fluxes first so we can apply the flux limit,
                    # we always apply this cut to the combined emission
                    f444w = unyt_array(
                        hdf[
                            "ImageObservedPhotometry/"
                            + spec.replace("agn_", "").replace("stellar_", "")
                            + "/JWST/NIRCam.F444W"
                        ][...],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    mask = np.where(f444w > flux_limit)
                    if np.sum(mask) == 0:
                        continue

                    # Apply the mask
                    f444w = f444w[mask]

                    # Get the indices
                    inds = hdf["Indices"][mask]
                    f115w = unyt_array(
                        hdf[f"ImageObservedPhotometry/{spec}/JWST/NIRCam.F115W"][mask],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f150w = unyt_array(
                        hdf[f"ImageObservedPhotometry/{spec}/JWST/NIRCam.F150W"][mask],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f200w = unyt_array(
                        hdf[f"ImageObservedPhotometry/{spec}/JWST/NIRCam.F200W"][mask],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f277w = unyt_array(
                        hdf[f"ImageObservedPhotometry/{spec}/JWST/NIRCam.F277W"][mask],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f356w = unyt_array(
                        hdf[f"ImageObservedPhotometry/{spec}/JWST/NIRCam.F356W"][mask],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    f444w = unyt_array(
                        hdf[f"ImageObservedPhotometry/{spec}/JWST/NIRCam.F444W"][mask],
                        "erg/s/cm**2/Hz",
                    ).to("nJy")
                    for filt in FILTER_CODES:
                        sizes[snap].setdefault(filt.split(".")[-1], []).extend([])
                        images[snap].setdefault(filt.split(".")[-1], []).extend(
                            hdf[f"Images/{spec}/{filt}"][mask]
                        )
                    comp = (
                        hdf[f"Apertures/0p4/{spec}/JWST/NIRCam.F444W"][mask]
                        / hdf[f"Apertures/0p2/{spec}/JWST/NIRCam.F444W"][mask]
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
                indices_by_region[snap].setdefault(reg, []).extend(inds)
                compactness[snap].extend(comp)

    # Convert the data to arrays
    for snap in fluxes.keys():
        for key in fluxes[snap].keys():
            fluxes[snap][key] = np.array(fluxes[snap][key]) * nJy
            sizes[snap][key] = np.array(sizes[snap][key])
            compactness[snap] = np.array(compactness[snap])
            for reg in REGIONS:
                if reg in indices_by_region[snap]:
                    indices_by_region[snap][reg] = np.array(
                        indices_by_region[snap][reg], dtype=int
                    )
                else:
                    indices_by_region[snap][reg] = np.array([], dtype=int)
        for filt in images[snap].keys():
            images[snap][filt] = np.array(images[snap][filt])

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
            compactness[snap] < 1.7,
        )
        masks[snap] = mask

    return fluxes, colors, red1, red2, sizes, masks, indices_by_region, images


def get_synth_spectra(synth_data_path, spec, cut_on=None):
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
    spectra = {}

    # If cut_on is none use spec
    if cut_on is None:
        cut_on = spec

    # Lood over regions
    for reg in REGIONS:
        # Loop over snapshots
        for snap in SNAPSHOTS:
            # Ensure a key exists for this snapshot
            spectra.setdefault(snap, [])

            # Skip files that don't exist
            if not os.path.exists(
                synth_data_path.replace("<region>", reg).replace("<snap>", snap)
            ):
                missing_path = synth_data_path.replace("<region>", reg).replace(
                    "<snap>", snap
                )
                print(f"File {missing_path} does not exist.")
                continue

            # Convert the flux limit to nJy
            flux_limit = apparent_mag_to_fnu(FLUX_LIMIT).to("nJy")

            # Get the fluxes we need
            with h5py.File(
                synth_data_path.replace("<region>", reg).replace("<snap>", snap),
                "r",
            ) as hdf:
                try:
                    # Get f444w fluxes first so we can apply the flux limit,
                    # we always apply this cut to the combined emission
                    f444w = unyt_array(
                        hdf[f"ImageObservedPhotometry/{cut_on}" "/JWST/NIRCam.F444W"][
                            ...
                        ],
                        "erg/s/cm**2/Hz",
                    )
                    mask = f444w > flux_limit
                    if np.sum(mask) == 0:
                        continue

                    # Get the spectra
                    s = unyt_array(
                        hdf[f"ObservedSpectra/{spec}"][mask, :],
                        "nJy",
                    ).to("nJy")
                except KeyError as e:
                    print(f"KeyError: {e} ({reg}/{snap})")
                    continue
                except OSError as e:
                    print(f"OSError: {e}")
                    continue
                except TypeError as e:
                    print(f"TypeError: {e}")
                    continue

                # Store the data
                spectra[snap].extend(s)

    # Convert the data to arrays
    for snap in spectra.keys():
        spectra[snap] = unyt_array(spectra[snap], "erg/s/cm**2/Hz").to("nJy")

    return spectra


def get_master_data(master_file_path, indices, key, get_weights=False):
    """
    Get the data from the master file for the given indices.

    Args:
        master_file_path (str): The path to the master file.
        indices (list): The indices of the galaxies to extract.
        key (str): The key of the data to extract.
        get_weights (bool): Whether to get the weights.

    Returns:
        dict: A dictionary containing the data for the galaxies.
    """
    if not get_weights:
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

        # Convert the data to arrays
        for snap in data.keys():
            data[snap] = np.array(data[snap])

        return data
    else:
        # Load the weights
        region_weights = np.loadtxt("data/weights.txt", usecols=8, delimiter=",")

        # Open the file
        with h5py.File(master_file_path, "r") as hdf:
            # Define the data dictionary
            data = {}
            weights = {}

            # Loop over regions
            for reg in REGIONS:
                # Loop over snapshots
                for snap in SNAPSHOTS:
                    # Ensure a key exists for this snapshot
                    data.setdefault(snap, [])
                    weights.setdefault(snap, [])

                    # Get the data we need
                    reg_data = hdf[f"{reg}/{snap}/Galaxy/{key}"][indices[snap][reg]]
                    data[snap].extend(reg_data)

                    # Get the weights
                    weights[snap].extend(
                        np.full(reg_data.shape[0], region_weights[int(reg)])
                    )

        # Convert the data to arrays
        for snap in data.keys():
            data[snap] = np.array(data[snap])
            weights[snap] = np.array(weights[snap])

        return data, weights


def get_masked_synth_data(synth_path, key, masks=None):
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
    if masks is not None:
        for snap in data.keys():
            data[snap] = np.array(data[snap])[masks[snap]]
    else:
        # Need to convert to arrays anyway
        for snap in data.keys():
            data[snap] = np.array(data[snap])

    return data


def get_galaxy_identifiers(master_file_path, indices):
    # Create dict to store galaxy identifiers
    gal_ids = {}

    # Open the file
    with h5py.File(master_file_path, "r") as hdf:
        # Loop over regions
        for reg in REGIONS:
            # Loop over snapshots
            for snap in SNAPSHOTS:
                # Ensure a key exists for this snapshot
                gal_ids.setdefault(snap, [])

                grp_nums = hdf[f"{reg}/{snap}/Galaxy/GroupNumber"][indices[snap][reg]]
                subgrp_nums = hdf[f"{reg}/{snap}/Galaxy/SubGroupNumber"][
                    indices[snap][reg]
                ]
                regs = np.full_like(grp_nums, int(reg))

                gal_ids[snap].extend(list(zip(regs, grp_nums, subgrp_nums)))

    # Convert the galaxy identifiers to an array
    for snap in gal_ids.keys():
        gal_ids[snap] = np.array(gal_ids[snap])

    return gal_ids


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
        bbox=dict(boxstyle="round,pad=0.3", fc="grey", ec="w", lw=1, alpha=0.7),
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
        bbox=dict(boxstyle="round,pad=0.3", fc="grey", ec="w", lw=1, alpha=0.7),
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
    savefig(fig, basename)


def plot_step_hist(
    outname, label, bins=50, normalise=False, weights=None, log=True, **kwargs
):
    """
    Plot a step histogram.

    Args:
        label (str): The x label of the histogram.
        bins (int): The number of bins to use.
        normalise (bool): Whether to normalise the histogram.
        **kwargs: The data to plot in label=data format.
    """
    # Create the figure
    fig, ax = plt.subplots()

    # Turn on the grid and set the axis below
    ax.grid(True)
    ax.set_axisbelow(True)

    # Remove all zeros
    okinds = {}
    for lab, data in kwargs.items():
        okinds[lab] = data != 0
        kwargs[lab] = data[data != 0]

    # If log we need to log the data and remove all 0s
    if log:
        for lab, data in kwargs.items():
            kwargs[lab] = np.log10(data)

    # Get the bins
    if isinstance(bins, int):
        # Get the min and max of the data
        _min = np.inf
        _max = -np.inf
        for data in kwargs.values():
            if len(data) == 0:
                continue
            _min = min(_min, np.min(data))
            _max = max(_max, np.max(data))

        # Exit if there is no data
        if _min == np.inf or _max == -np.inf:
            return

        # Set the bins
        bins = np.linspace(_min, _max, bins)

    # Loop over the data
    for lab, data in kwargs.items():
        if weights is None:
            ws = np.ones_like(data)
        else:
            ws = weights[lab][okinds[lab]]

        ax.hist(
            data,
            bins=bins,
            histtype="step",
            label=lab,
            linewidth=1.5,
            density=normalise,
            weights=ws,
        )

    # Set the labels
    ax.set_xlabel(label)
    if normalise:
        ax.set_ylabel("PDF")
    else:
        ax.set_ylabel("$N$")

    # Add the legend
    ax.legend()

    # Save the figure
    if normalise:
        savefig(
            fig,
            f"{outname}_normed",
        )
    else:
        savefig(
            fig,
            f"{outname}",
        )
    plt.close(fig)


def sort_data_recursive(data, sinds):
    """
    Sort a dictionary recursively.

    Args:
        data (dict): The data to sort.
        sinds (dict): The sorted indices.
    """
    # If the data isn't a dictionary just return the sorted data
    if not isinstance(data, dict):
        data = np.array(data)
        return data[sinds]

    # Loop over the data
    sorted_data = {}
    for k, v in data.items():
        sorted_data[k] = sort_data_recursive(v, sinds)

    return sorted_data


def write_dataset_recursive(hdf, data, key, units="dimensionless"):
    """
    Write a dictionary to an HDF5 file recursively.

    Args:
        hdf (h5py.File): The HDF5 file to write to.
        data (dict): The data to write.
        key (str): The key to write the data to.
        units (str): The units of the data.
    """
    # If the data isn't a dictionary just write the dataset
    if not isinstance(data, dict):
        _print(f"Writing {key}")
        dset = hdf.create_dataset(key, data=data)
        dset.attrs["Units"] = units
        return

    # Loop over the data
    for k, v in data.items():
        write_dataset_recursive(hdf, v, f"{key}/{k}", units=units)


def _combine_data_recursive(rank_data, out_dict, key=None):
    """
    Combine data from multiple processors recursively.

    Args:
        rank_data (dict): The data from the current processor.
        out_dict (dict): The dictionary to write the data to.
    """
    if not isinstance(rank_data, dict):
        out_dict.extend(rank_data)
        return out_dict

    # Loop over the data
    for k, v in rank_data.items():
        if isinstance(v, list):
            out_dict[k] = _combine_data_recursive(v, out_dict.get(k, []), k)
        else:
            out_dict[k] = _combine_data_recursive(v, out_dict.get(k, {}), k)

    return out_dict


def combine_distributed_data(distributed_data):
    """
    Combine data from multiple processors.

    Args:
        distributed_data (dict): The data from each processor.
        out_dict (dict): The dictionary to write the data to.
    """
    # Setup output, if we have a list of lists we only need to combine them,
    # otherwise we need to recusively combine the data in dicts
    if isinstance(distributed_data[0], list):
        # Just check for empty lists and remove them
        distributed_data = [d for d in distributed_data if len(d) > 0]
        if len(distributed_data) == 0:
            return np.array([])
        return np.concatenate(distributed_data)

    # Ok, we need to initialise the output dict
    out_dict = {}

    # Loop over the list of data from ranks
    for rank_data in distributed_data:
        out_dict = _combine_data_recursive(rank_data, out_dict)

    return out_dict


def recursive_gather(data, comm, root=0):
    """
    Recursively collect data from all ranks onto the root.

    This function is effectively just comm.gather but will ensure the
    gather is performed at the root of any passed dictionaries to avoid
    overflows and any later recursive concatenation issues.

    Args:
        data (any): The data to gather.
        comm (mpi.Comm): The MPI communicator.
        root (int): The root rank to gather data to.
    """
    # If we don't have a dict, just gather the data straight away
    if not isinstance(data, dict):
        collected_data = comm.gather(data, root=root)
        if comm.rank == root:
            collected_data = [l for l in collected_data if len(l) > 0]
            if len(collected_data) > 0:
                return np.concatenate(collected_data)
            else:
                return []
        else:
            return []

    # Recurse through the whole dict communicating an lists or
    # arrays we hit along the way
    def _gather(d, comm, root):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, (list, np.ndarray)):
                collected_data = comm.gather(v, root=root)
                if comm.rank == root:
                    collected_data = [l for l in collected_data if len(l) > 0]
                    if len(collected_data) > 0:
                        new_d[k] = np.concatenate(collected_data)
                    else:
                        new_d[k] = []
                else:
                    new_d[k] = []
            elif isinstance(v, dict):
                new_d[k] = _gather(v, comm, root)
            else:
                raise ValueError(f"Unexpected type {type(v)}")
        return new_d

    return _gather(data, comm, root)
