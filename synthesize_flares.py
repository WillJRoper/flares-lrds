"""A script to derive synthetic observations for FLARES."""

import argparse
import time
import os
import multiprocessing as mp
import numpy as np
import h5py
from unyt import Gyr, Mpc, Msun
from astropy.cosmology import Planck15 as cosmo

from synthesizer.particle import Stars, Gas
from synthesizer.particle import Galaxy
from synthesizer.filters import FilterCollection
from synthesizer import Grid
from synthesizer.kernel_functions import Kernel

from stellar_emission_model import FLARESLOSEmission


def _get_galaxy(gal_ind, master_file_path, reg, snap, z):
    """
    Get a galaxy from the master file.

    Args:
        gal_ind (int): The index of the galaxy to get.
    """
    # Get the galaxy data we need from the master file
    with h5py.File(master_file_path, "r") as hdf:
        reg_grp = hdf[reg]
        snap_grp = reg_grp[snap]
        gal_grp = snap_grp["Galaxy"]
        part_grp = snap_grp["Particle"]

        # Get this galaxy's beginning and ending indices for stars
        s_len = gal_grp["S_Length"][...]
        start = np.sum(s_len[:gal_ind])
        end = np.sum(s_len[: gal_ind + 1])

        # Get this galaxy's beginning and ending indices for gas
        g_len = gal_grp["G_Length"][...]
        start_gas = np.sum(g_len[:gal_ind])
        end_gas = np.sum(g_len[: gal_ind + 1])

        # Get the star data
        star_pos = part_grp["S_Coordinates"][:, start:end].T * Mpc
        star_mass = part_grp["S_Mass"][start:end] * Msun * 10**10
        star_init_mass = part_grp["S_MassInitial"][start:end] * Msun * 10**10
        star_age = part_grp["S_Age"][start:end] * Gyr
        star_met = part_grp["S_Z_smooth"][start:end]
        star_sml = part_grp["S_sml"][start:end] * Mpc

        # Get the gas data
        gas_pos = part_grp["G_Coordinates"][:, start_gas:end_gas].T * Mpc
        gas_mass = part_grp["G_Mass"][start_gas:end_gas] * Msun * 10**10
        gas_met = part_grp["G_Z_smooth"][start_gas:end_gas]
        gas_sml = part_grp["G_sml"][start_gas:end_gas] * Mpc

    # Early exist if there are fewer than 100 baryons
    if star_mass.size < 100:
        return None

    gal = Galaxy(
        name=f"{reg}/{snap}/{gal_ind}",
        redshift=z,
        stars=Stars(
            initial_masses=star_init_mass,
            current_masses=star_mass,
            ages=star_age,
            metallicities=star_met,
            redshift=z,
            coordinates=star_pos,
            smoothing_lengths=star_sml,
            centre=star_pos.mean(axis=0),
        ),
        gas=Gas(
            masses=gas_mass,
            metallicities=gas_met,
            redshift=z,
            coordinates=gas_pos,
            smoothing_lengths=gas_sml,
            centre=gas_pos.mean(axis=0),
        ),
    )

    # Attach the extra tau_v we need for nebular attenuation around young
    # stars
    gal.stars.young_tau_v = star_met / 0.01

    return gal


def get_flares_galaxies(master_file_path, region, snap, nthreads):
    """
    Get Galaxy objects for FLARES galaxies.

    Args:
        master_file_path (str): The path to the master file.
        region (int): The region to use.
        snap (str): The snapshot to use.
        filter_collection (FilterCollection): The filter collection to use.
        emission_model (StellarEmissionModel): The emission model to use.
    """
    # Get the region tag
    reg = str(region).zfill(2)

    # Get redshift from the snapshot tag
    z = float(snap.split("_")[-1].replace("z", "").replace("p", "."))

    # How many galaxies are there?
    with h5py.File(master_file_path, "r") as hdf:
        reg_grp = hdf[reg]
        snap_grp = reg_grp[snap]
        gal_grp = snap_grp["Galaxy"]
        n_gals = len(gal_grp["S_Length"])

    # Early exist if there are no galaxies
    if n_gals == 0:
        return []

    # Prepare the arguments for each galaxy
    args = [
        (gal_ind, master_file_path, reg, snap, z) for gal_ind in range(n_gals)
    ]

    # Get all the galaxies using multiprocessing
    with mp.Pool(nthreads) as pool:
        galaxies = pool.starmap(_get_galaxy, args)

    # Remove any Nones
    galaxies = [gal for gal in galaxies if gal is not None]

    return galaxies


def get_flares_filters(filepath):
    """Get the filter collection."""
    # Check if the filter collection file already exists
    if os.path.exists(filepath):
        filters = FilterCollection(path=filepath)
    else:
        # Define the list of filter IDs
        filter_codes = [
            "JWST/NIRCam.F115W",
            "JWST/NIRCam.F150W",
            "JWST/NIRCam.F200W",
            "JWST/NIRCam.F277W",
            "JWST/NIRCam.F356W",
            "JWST/NIRCam.F444W",
        ]

        # Create the FilterCollection
        filters = FilterCollection(filter_codes=filter_codes)

        # Write the filter collection
        filters.write_filters(path=filepath)

    return filters


def get_grid(grid_name, grid_dir, filters):
    """Get a Synthesizer Grid."""
    return Grid(
        grid_name,
        grid_dir,
        # filters=filters,
    )


def get_emission_model(grid, fesc=0.0, fesc_ly_alpha=1.0):
    """Get a StellarEmissionModel."""
    return FLARESLOSEmission(grid, fesc=fesc, fesc_ly_alpha=fesc_ly_alpha)


def get_kernel():
    """Get a Kernel."""
    return Kernel()


def get_spectra(gal, emission_model, kern, nthreads):
    """Get the spectra for a galaxy."""
    # Get the los tau_v
    gal.calculate_los_tau_v(
        kappa=0.0795,
        kernel=kern.get_kernel(),
        force_loop=False,
    )

    # Get the spectra
    gal.stars.get_particle_spectra(
        emission_model,
        nthreads=nthreads,
    )

    return gal


def get_photometry(gal, filters, spectra_key, cosmo):
    """Get the photometry for a galaxy."""
    # Get the flux
    gal.stars.particle_spectra[spectra_key].get_fnu(cosmo, gal.redshift)

    # Get the photometry
    phot = gal.stars.particle_spectra[spectra_key].get_photo_fluxes(filters)

    return phot


def get_image():
    pass


# Define the snapshot tags
snapshots = [
    "005_z010p000",
    "006_z009p000",
    "007_z008p000",
    "008_z007p000",
    "009_z006p000",
    "010_z005p000",
]

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Derive synthetic observations for FLARES."
    )

    # Add the arguments
    parser.add_argument(
        "master_file_path",
        type=str,
        help="The path to the master file.",
    )
    parser.add_argument(
        "--grid",
        type=str,
        help="The path to the grid.",
    )
    parser.add_argument(
        "--grid-dir",
        type=str,
        help="The directory to save the grid.",
    )
    parser.add_argument(
        "--region",
        type=int,
        help="The region to use.",
    )
    parser.add_argument(
        "--snap",
        type=int,
        help="The snapshot to use.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        help="The number of threads to use.",
    )

    # Parse the arguments
    args = parser.parse_args()
    path = args.master_file_path
    grid_name = args.grid
    grid_dir = args.grid_dir
    region = args.region
    snap = snapshots[args.snap]
    nthreads = args.nthreads

    start = time.time()

    # Get the filters
    filt_start = time.time()
    filters = get_flares_filters("lrd_filters.hdf5")
    filt_end = time.time()
    print(f"Getting filters took {filt_end - filt_start:.2f} seconds.")

    # Get the grid
    grid_start = time.time()
    grid = get_grid(grid_name, grid_dir, filters)
    grid_end = time.time()

    # Get the emission model
    start_emission = time.time()
    emission_model = get_emission_model(grid)
    end_emission = time.time()
    print(
        f"Getting the emission model took "
        f"{end_emission - start_emission:.2f} seconds."
    )

    # Get the kernel
    start_kernel = time.time()
    kern = get_kernel()
    end_kernel = time.time()

    # Get the galaxies
    read_start = time.time()
    galaxies = get_flares_galaxies(
        path,
        region,
        snap,
        nthreads,
    )
    read_end = time.time()
    print(
        f"Creating {len(galaxies)} galaxies took "
        f"{read_end - read_start:.2f} seconds."
    )

    # Get the spectra
    spectra_start = time.time()
    galaxies = [
        get_spectra(gal, emission_model, kern, nthreads) for gal in galaxies
    ]
    spectra_end = time.time()
    print(f"Getting spectra took {spectra_end - spectra_start:.2f} seconds.")

    # Get the photometry
    phot_start = time.time()
    phot = []
    for gal in galaxies:
        p = get_photometry(gal, filters, "reprocessed", cosmo)
        print(p)
        phot.append(p)
    phot_end = time.time()
    print(f"Getting photometry took {phot_end - phot_start:.2f} seconds.")

    end = time.time()
    print(f"Total time: {end - start:.2f} seconds.")
