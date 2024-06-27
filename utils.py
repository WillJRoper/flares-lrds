"""A module containing the definition of helpful functions."""
import h5py
import numpy as np


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


def get_sizes_and_mdot(master_file_path):
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


def get_fluxes_colors(synth_data_path, spec):
    """
    Get the fluxes and colors of the galaxies in the simulation.

    Args:
        synth_data_path (str): The path to the synthetic data.
        spec (str): The spectral synthesis model to use.

    Returns:
        dict: A dictionary containing the fluxes of the galaxies.
        dict: A dictionary containing the colors of the galaxies.
        dict: A dictionary containing the red1 mask of the galaxies.
        dict: A dictionary containing the red2 mask of the galaxies.
    """
    # Define containers for the data
    fluxes = {}

    # Loop over regions
    for reg in REGIONS:
        # Loop over snapshots
        for snap in SNAPSHOTS:
            # Ensure a key exists for this snapshot
            fluxes.setdefault(snap, {})

            # Get the fluxes we need
            with h5py.File(
                synth_data_path.replace("<region>", reg).replace(
                    "<snap>", snap
                ),
                "r",
            ) as hdf:
                try:
                    f115w = hdf[
                        f"ObservedPhotometry/{spec}/JWST/NIRCam.F115W"
                    ][...]
                    f150w = hdf[
                        f"ObservedPhotometry/{spec}/JWST/NIRCam.F150W"
                    ][...]
                    f200w = hdf[
                        f"ObservedPhotometry/{spec}/JWST/NIRCam.F200W"
                    ][...]
                    f277w = hdf[
                        f"ObservedPhotometry/{spec}/JWST/NIRCam.F277W"
                    ][...]
                    f356w = hdf[
                        f"ObservedPhotometry/{spec}/JWST/NIRCam.F356W"
                    ][...]
                    f444w = hdf[
                        f"ObservedPhotometry/{spec}/JWST/NIRCam.F444W"
                    ][...]
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

    # Convert the data to arrays
    for snap in fluxes.keys():
        for key in fluxes[snap].keys():
            fluxes[snap][key] = np.array(fluxes[snap][key])

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

    return fluxes, colors, red1, red2
