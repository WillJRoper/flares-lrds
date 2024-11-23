"""A pipeline for generating synthetic observations from FLARES."""

import argparse
import os
import warnings
from functools import partial

import h5py
import numpy as np
from astropy.cosmology import Planck15 as cosmo
from mpi4py import MPI as mpi
from pathos.multiprocessing import ProcessingPool as Pool
from synthesizer.conversions import angular_to_spatial_at_z
from synthesizer.grid import Grid
from synthesizer.instruments import InstrumentCollection
from synthesizer.kernel_functions import Kernel
from synthesizer.particle import BlackHoles, Galaxy, Gas, Stars
from synthesizer.survey import Survey
from unyt import Gyr, Mpc, Msun, angstrom, arcsecond, km, kpc, s, yr

from combined_emission_model import FLARESLOSCombinedEmission
from utils import (
    SPECTRA_KEYS,
)

# Silence warnings (only because we now what we're doing)
warnings.filterwarnings("ignore")

# Msun needs to be respected
Msun = Msun.in_base("galactic")


def _get_galaxy(gal_index, master_file_path, snap):
    """
    Get a galaxy from the master file.

    Args:
        gal_index (int): The index of the galaxy to get.
        master_file_path (str): The path to the master file.
        snap (str): The snapshot to use.
    """
    # Get the redshift from the tag
    z = float(snap.split("z")[-1].replace("p", "."))

    # Get the galaxy data we need from the master file
    with h5py.File(master_file_path, "r") as hdf:
        # What region are we in? We'll find this by looping over regions until
        # we find the one that contains the galaxy index
        regions = list(hdf.keys())
        reg_ind = 0
        reg_ngal = len(hdf[regions[reg_ind]][snap]["Galaxy"]["S_Length"])
        while gal_index >= reg_ngal:
            gal_index -= reg_ngal
            reg_ind += 1
            reg_ngal = len(hdf[regions[reg_ind]][snap]["Galaxy"]["S_Length"])
        reg = regions[reg_ind]

        # Get the weight for this region
        region_weight = np.loadtxt(
            "data/weights.txt",
            usecols=8,
            delimiter=",",
        )[int(reg)]

        # Unpack the groups
        reg_grp = hdf[reg]
        snap_grp = reg_grp[snap]
        gal_grp = snap_grp["Galaxy"]
        part_grp = snap_grp["Particle"]

        # Get the group and subgrp ids
        group_id = gal_grp["GroupNumber"][gal_index]
        subgrp_id = gal_grp["SubGroupNumber"][gal_index]

        # Get this galaxy's beginning and ending indices for stars
        s_len = gal_grp["S_Length"][...]
        start = np.sum(s_len[:gal_index])
        end = np.sum(s_len[: gal_index + 1])

        # Get this galaxy's beginning and ending indices for gas
        g_len = gal_grp["G_Length"][...]
        start_gas = np.sum(g_len[:gal_index])
        end_gas = np.sum(g_len[: gal_index + 1])

        # Get this galaxy's beginning and ending indices for black holes
        bh_len = gal_grp["BH_Length"][...]
        start_bh = np.sum(bh_len[:gal_index])
        end_bh = np.sum(bh_len[: gal_index + 1])

        # Get the star data
        star_pos = part_grp["S_Coordinates"][:, start:end].T / (1 + z) * Mpc
        star_mass = part_grp["S_Mass"][start:end] * Msun * 10**10
        star_init_mass = part_grp["S_MassInitial"][start:end] * Msun * 10**10
        star_age = part_grp["S_Age"][start:end] * Gyr
        star_met = part_grp["S_Z_smooth"][start:end]
        star_sml = part_grp["S_sml"][start:end] * Mpc
        star_vel = part_grp["S_Vel"][:, start:end].T * km / s

        # Get the gas data
        gas_pos = part_grp["G_Coordinates"][:, start_gas:end_gas].T / (1 + z) * Mpc
        gas_mass = part_grp["G_Mass"][start_gas:end_gas] * Msun * 10**10
        gas_met = part_grp["G_Z_smooth"][start_gas:end_gas]
        gas_sml = part_grp["G_sml"][start_gas:end_gas] * Mpc
        gas_vel = part_grp["G_Vel"][:, start_gas:end_gas].T * km / s

        # Get the black hole data
        bh_pos = part_grp["BH_Coordinates"][:, start_bh:end_bh].T / (1 + z) * Mpc
        bh_mass = part_grp["BH_Mass"][start_bh:end_bh] * Msun * 10**10
        bh_mdot = (
            part_grp["BH_Mdot"][start_bh:end_bh]
            * (6.445909132449984 * 10**23)  # Unit conversion issue, need this
            / 0.6777  # divide by magic extra h (its really there!)
            * Msun
            / yr
        )

        # Get the centre of potential
        centre = gal_grp["COP"][:].T[gal_index, :] / (1 + z) * Mpc

        # Compute the angular radii of each star in arcseconds
        radii = (np.linalg.norm(star_pos - centre, axis=1)).to("kpc")
        gradii = (np.linalg.norm(gas_pos - centre, axis=1)).to("kpc")
        bhradii = (np.linalg.norm(bh_pos - centre, axis=1)).to("kpc")
        star_ang_rad = radii.value * cosmo.arcsec_per_kpc_proper(z).value * arcsecond

    # Define a mask to get a 30 kpc aperture
    mask = radii < 30 * kpc
    gmask = gradii < 30 * kpc
    bhmask = bhradii < 30 * kpc

    # Early exit if there are fewer than 100 stars
    # if np.sum(mask) < 100:
    if np.sum(mask) > 105 or np.sum(mask) < 100:
        return None

    gal = Galaxy(
        name=f"{reg}_{snap}_{group_id}_{subgrp_id}",
        redshift=z,
        master_index=gal_index,
        region=reg,
        grp_id=group_id,
        subgrp_id=subgrp_id,
        weight=region_weight,
        stars=Stars(
            initial_masses=star_init_mass[mask],
            current_masses=star_mass[mask],
            ages=star_age[mask],
            metallicities=star_met[mask],
            redshift=z,
            coordinates=star_pos[mask, :],
            velocities=star_vel[mask, :],
            smoothing_lengths=star_sml[mask],
            centre=centre,
            young_tau_v=star_met[mask] / 0.01,
            angular_radii=star_ang_rad[mask],
            radii=radii[mask].value,
        ),
        gas=Gas(
            masses=gas_mass[gmask],
            metallicities=gas_met[gmask],
            redshift=z,
            coordinates=gas_pos[gmask, :],
            smoothing_lengths=gas_sml[gmask],
            centre=centre,
            velocities=gas_vel[gmask, :],
        ),
        black_holes=BlackHoles(
            masses=bh_mass[bhmask],
            accretion_rates=bh_mdot[bhmask],
            coordinates=bh_pos[bhmask, :],
            redshift=z,
            centre=centre,
        ),
    )

    # Calculate the DTM, we'll need it later
    gal.dust_to_metal_vijayan19()

    return gal


def partition_galaxies(galaxy_weights):
    """Partition the galaxies between the MPI processes."""
    # Get the number of galaxies and the number of processes
    ngals = len(galaxy_weights)
    nranks = mpi.COMM_WORLD.Get_size()
    rank = mpi.COMM_WORLD.Get_rank()

    # Create and index array
    indices = np.arange(ngals)

    # Assign the galaxies to balance the weights
    weight_on_rank = np.zeros(nranks)
    inds_on_rank = {i: [] for i in range(nranks)}
    for i in range(ngals):
        # Find the rank with the smallest weight
        min_rank = np.argmin(weight_on_rank)
        weight_on_rank[min_rank] += galaxy_weights[i]
        inds_on_rank[min_rank].append(indices[i])

    return inds_on_rank[rank]


def load_galaxies(master_file_path, snap, indices, nthreads=1):
    """Load the galaxies into memory."""
    # Load the galaxies
    if nthreads > 1:
        with Pool(nthreads) as pool:
            galaxies = pool.map(
                partial(
                    _get_galaxy,
                    master_file_path=master_file_path,
                    snap=snap,
                ),
                indices,
            )
    else:
        galaxies = []
        for i, gal_index in enumerate(indices):
            gal = _get_galaxy(gal_index, master_file_path, snap)
            galaxies.append(gal)

    # Remove any galaxies that are None
    galaxies = [gal for gal in galaxies if gal is not None]

    return galaxies


def get_emission_model(
    grid_name,
    grid_dir,
    fesc=0.0,
    fesc_ly_alpha=1.0,
    agn_template_file="vandenberk_agn_template.txt",
    save_spectra=SPECTRA_KEYS,
):
    """Get the emission model to use for the observations."""
    grid = Grid(
        grid_name,
        grid_dir,
        lam_lims=(900 * angstrom, 6 * 10**5 * angstrom),
    )
    model = FLARESLOSCombinedEmission(
        agn_template_file,
        grid,
        fesc=fesc,
        fesc_ly_alpha=fesc_ly_alpha,
    )

    # Limit the spectra to be saved
    # model.save_spectra(*save_spectra)

    return model


def get_stars_1d_velocity_dispersion(gal):
    """Get the 1D velocity dispersion of the stars in the galaxy."""
    return (
        np.array(
            [
                np.std(gal.stars._velocities[:, 0], ddof=0),
                np.std(gal.stars._velocities[:, 1], ddof=0),
                np.std(gal.stars._velocities[:, 2], ddof=0),
            ]
        )
        * gal.stars.velocities.units
    )


def get_gas_1d_velocity_dispersion(gal):
    """Get the 1D velocity dispersion of the gas in the galaxy."""
    return (
        np.array(
            [
                np.std(gal.gas._velocities[:, 0], ddof=0),
                np.std(gal.gas._velocities[:, 1], ddof=0),
                np.std(gal.gas._velocities[:, 2], ddof=0),
            ]
        )
        * gal.gas.velocities.units
    )


def get_stars_3d_velocity_dispersion(gal):
    """Get the 3D velocity dispersion of the stars in the galaxy."""
    return (
        np.std(np.sqrt(np.sum(gal.stars._velocities**2, axis=1)), ddof=0)
        * gal.stars.velocities.units
    )


def get_gas_3d_velocity_dispersion(gal):
    """Get the 3D velocity dispersion of the gas in the galaxy."""
    return (
        np.std(np.sqrt(np.sum(gal.gas._velocities**2, axis=1)), ddof=0)
        * gal.gas.velocities.units
    )


def get_colors_and_lrd_flags(gal, cosmo, nthreads):
    """
    Get the colors and LRD flags for the galaxy and it's components.

    The LRD status is defined by a set of colors and a compactification metric
    drived from a 0.2" and 0.4" aperture in the F444W filter. The colors are
    defined as:

        F115W - F150W < 0.8
        F200W - F277W > 0.7
        F200W - F356W > 1.0

        or

        F150W - F200W < 0.8
        F277W - F356W > 0.6
        F277W - F444W > 0.7

    Args:
        gal (Galaxy): The galaxy to get the colors and LRD flags for.
    """
    # Define a dictionary to store the results in (galaxy level results are
    # stored in the "Galaxy" key)
    results = {"Stars": {}, "BlackHoles": {}}

    # Compute spatial apertures
    ang_apertures = np.array([0.2, 0.4]) * arcsecond
    kpc_apertures = angular_to_spatial_at_z(ang_apertures, cosmo, gal.redshift)

    # Compute aperture photometry
    results["AperturePhotometry"] = {}
    results["Stars"]["AperturePhotometry"] = {}
    results["BlackHoles"]["AperturePhotometry"] = {}
    for inst, d in gal.images_psf_fnu.items():
        for spec_type, imgs in d.items():
            results["AperturePhotometry"][spec_type] = {}
            for filt, img in imgs.items():
                results["AperturePhotometry"][spec_type][filt] = {}
                for i, ang_ap in enumerate(ang_apertures):
                    results["AperturePhotometry"][spec_type][filt][
                        f"Aperture_{kpc_apertures[i]}".replace(".", "p")
                    ] = img.get_signal_in_aperture(
                        kpc_apertures[i].to("Mpc"),
                        nthreads=nthreads,
                    )
    for inst, d in gal.stars.images_psf_fnu.items():
        for spec_type, imgs in d.items():
            results["Stars"]["AperturePhotometry"][spec_type] = {}
            for filt, img in imgs.items():
                results["Stars"]["AperturePhotometry"][spec_type][filt] = {}
                for i, ang_ap in enumerate(ang_apertures):
                    results["Stars"]["AperturePhotometry"][spec_type][filt][
                        f"Aperture_{kpc_apertures[i]}".replace(".", "p")
                    ] = img.get_signal_in_aperture(
                        kpc_apertures[i].to("Mpc"),
                        nthreads=nthreads,
                    )
    for inst, d in gal.black_holes.images_psf_fnu.items():
        for spec_type, imgs in d.items():
            results["BlackHoles"]["AperturePhotometry"][spec_type] = {}
            for filt, img in imgs.items():
                results["BlackHoles"]["AperturePhotometry"][spec_type][filt] = {}
                for i, ang_ap in enumerate(ang_apertures):
                    results["BlackHoles"]["AperturePhotometry"][spec_type][filt][
                        f"Aperture_{kpc_apertures[i]}".replace(".", "p")
                    ] = img.get_signal_in_aperture(
                        kpc_apertures[i].to("Mpc"),
                        nthreads=nthreads,
                    )

    # Compute the compactness criterion
    results["Compactness"] = {}
    results["Stars"]["Compactness"] = {}
    results["BlackHoles"]["Compactness"] = {}
    for spec_type in results["AperturePhotometry"].keys():
        if spec_type == "Stars" or spec_type == "Blackholes":
            continue
        results["Compactness"][spec_type] = (
            results["AperturePhotometry"][spec_type]["JWST/NIRCam.F444W"][
                "Aperture_0p4"
            ]
            / results["AperturePhotometry"][spec_type]["JWST/NIRCam.F444W"][
                "Aperture_0p2"
            ]
        )
    for spec_type in results["Stars"]["AperturePhotometry"].keys():
        results["Stars"]["Compactness"][spec_type] = (
            results["Stars"]["AperturePhotometry"][spec_type]["JWST/NIRCam.F444W"][
                "Aperture_0p4"
            ]
            / results["Stars"]["AperturePhotometry"][spec_type]["JWST/NIRCam.F444W"][
                "Aperture_0p2"
            ]
        )
    for spec_type in results["BlackHoles"]["AperturePhotometry"].keys():
        results["BlackHoles"]["Compactness"][spec_type] = (
            results["BlackHoles"]["AperturePhotometry"][spec_type]["JWST/NIRCam.F444W"][
                "Aperture_0p4"
            ]
            / results["BlackHoles"]["AperturePhotometry"][spec_type][
                "JWST/NIRCam.F444W"
            ]["Aperture_0p2"]
        )

    # Do the galaxy level colors
    results["Colors"] = {}
    results["Stars"]["Colors"] = {}
    results["BlackHoles"]["Colors"] = {}
    for spec_type, phot in gal.photo_fnu.items():
        results["Colors"]["F115W_F150W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F115W"] / phot["JWST/NIRCam.F150W"]
        )
        results["Colors"]["F150W_F200W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F150W"] / phot["JWST/NIRCam.F200W"]
        )
        results["Colors"]["F200W_F277W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F200W"] / phot["JWST/NIRCam.F277W"]
        )
        results["Colors"]["F200W_F356W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F200W"] / phot["JWST/NIRCam.F356W"]
        )
        results["Colors"]["F277W_F356W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F277W"] / phot["JWST/NIRCam.F356W"]
        )
        results["Colors"]["F277W_F444W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F277W"] / phot["JWST/NIRCam.F444W"]
        )
    for spec_type, phot in gal.stars.photo_fnu.items():
        results["Stars"]["Colors"]["F115W_F150W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F115W"] / phot["JWST/NIRCam.F150W"]
        )
        results["Stars"]["Colors"]["F150W_F200W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F150W"] / phot["JWST/NIRCam.F200W"]
        )
        results["Stars"]["Colors"]["F200W_F277W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F200W"] / phot["JWST/NIRCam.F277W"]
        )
        results["Stars"]["Colors"]["F200W_F356W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F200W"] / phot["JWST/NIRCam.F356W"]
        )
        results["Stars"]["Colors"]["F277W_F356W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F277W"] / phot["JWST/NIRCam.F356W"]
        )
        results["Stars"]["Colors"]["F277W_F444W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F277W"] / phot["JWST/NIRCam.F444W"]
        )
    for spec_type, phot in gal.black_holes.photo_fnu.items():
        results["BlackHoles"]["Colors"]["F115W_F150W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F115W"] / phot["JWST/NIRCam.F150W"]
        )
        results["BlackHoles"]["Colors"]["F150W_F200W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F150W"] / phot["JWST/NIRCam.F200W"]
        )
        results["BlackHoles"]["Colors"]["F200W_F277W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F200W"] / phot["JWST/NIRCam.F277W"]
        )
        results["BlackHoles"]["Colors"]["F200W_F356W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F200W"] / phot["JWST/NIRCam.F356W"]
        )
        results["BlackHoles"]["Colors"]["F277W_F356W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F277W"] / phot["JWST/NIRCam.F356W"]
        )
        results["BlackHoles"]["Colors"]["F277W_F444W"] = -2.5 * np.log10(
            phot["JWST/NIRCam.F277W"] / phot["JWST/NIRCam.F444W"]
        )

    # Define the LRD flags
    results["LRDFlag"] = {}
    results["Stars"]["LRDFlag"] = {}
    results["BlackHoles"]["LRDFlag"] = {}
    for spec_type in results["Compactness"].keys():
        comp_mask = results["Compactness"][spec_type] < 1.7
        mask1 = np.logical_and(
            results["Colors"][spec_type]["F115W_F150W"] < 0.8,
            results["Colors"][spec_type]["F200W_F277W"] > 0.7,
        )
        mask1 = np.logical_and(mask1, results["Colors"][spec_type]["F200W_F356W"] > 1.0)
        mask2 = np.logical_and(
            results["Colors"][spec_type]["F150W_F200W"] < 0.8,
            results["Colors"][spec_type]["F277W_F356W"] > 0.6,
        )
        mask2 = np.logical_and(mask2, results["Colors"][spec_type]["F277W_F444W"] > 0.7)
        results["LRDFlag"][spec_type] = np.logical_and(
            comp_mask, np.logical_or(mask1, mask2)
        )
    for spec_type in results["Stars"]["Compactness"].keys():
        comp_mask = results["Stars"]["Compactness"][spec_type] < 1.7
        mask1 = np.logical_and(
            results["Stars"]["Colors"][spec_type]["F115W_F150W"] < 0.8,
            results["Stars"]["Colors"][spec_type]["F200W_F277W"] > 0.7,
        )
        mask1 = np.logical_and(
            mask1, results["Stars"]["Colors"][spec_type]["F200W_F356W"] > 1.0
        )
        mask2 = np.logical_and(
            results["Stars"]["Colors"][spec_type]["F150W_F200W"] < 0.8,
            results["Stars"]["Colors"][spec_type]["F277W_F356W"] > 0.6,
        )
        mask2 = np.logical_and(
            mask2, results["Stars"]["Colors"][spec_type]["F277W_F444W"] > 0.7
        )
        results["Stars"]["LRDFlag"][spec_type] = np.logical_and(
            comp_mask, np.logical_or(mask1, mask2)
        )
    for spec_type in results["BlackHoles"]["Compactness"].keys():
        comp_mask = results["BlackHoles"]["Compactness"][spec_type] < 1.7
        mask1 = np.logical_and(
            results["BlackHoles"]["Colors"][spec_type]["F115W_F150W"] < 0.8,
            results["BlackHoles"]["Colors"][spec_type]["F200W_F277W"] > 0.7,
        )
        mask1 = np.logical_and(
            mask1, results["BlackHoles"]["Colors"][spec_type]["F200W_F356W"] > 1.0
        )
        mask2 = np.logical_and(
            results["BlackHoles"]["Colors"][spec_type]["F150W_F200W"] < 0.8,
            results["BlackHoles"]["Colors"][spec_type]["F277W_F356W"] > 0.6,
        )
        mask2 = np.logical_and(
            mask2, results["BlackHoles"]["Colors"][spec_type]["F277W_F444W"] > 0.7
        )
        results["BlackHoles"]["LRDFlag"][spec_type] = np.logical_and(
            comp_mask, np.logical_or(mask1, mask2)
        )

    return results


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
        "--snap-ind",
        type=int,
        help="The snapshot to use.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        help="The number of threads to use.",
    )

    # Get MPI info
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse the arguments
    args = parser.parse_args()
    path = args.master_file_path
    grid_name = args.grid
    grid_dir = args.grid_dir
    snap = snapshots[args.snap_ind]
    nthreads = args.nthreads

    # Define the output path
    outpath = f"data/combined_{snap}.hdf5"

    # If the output already exists just exit
    if os.path.exists(outpath):
        print(f"Output file {outpath} already exists.")
        exit(0)

    # Get the number of galaxies we'll run with and the galaxy weights (the
    # number of stars in each galaxy)
    gal_weights = []
    with h5py.File(path, "r") as hdf:
        for reg in hdf.keys():
            gal_weights.extend(
                hdf[reg][snap]["Galaxy"]["S_Length"][:]
                # + hdf[reg][snap]["Galaxy"]["G_Length"][:]
            )
    ngals = len(gal_weights)
    gal_weights = np.array(gal_weights)

    # Sanitise out the galaxies with fewer than 100 baryons, we'll remove these
    # anyway
    okinds = np.logical_and(gal_weights < 100, gal weights > 105)
    # okinds = gal_weights < 100
    gal_weights[okinds] = 0

    # Get the SPH kernel
    kernel_data = Kernel().get_kernel()

    # Set up the survey
    survey = Survey(
        emission_model=get_emission_model(
            grid_name,
            grid_dir,
            fesc=0.0,
            fesc_ly_alpha=1.0,
        ),
        instruments=InstrumentCollection(
            filepath=f"flares_lrd_instruments_{snap}.hdf5"
        ),
        nthreads=nthreads,
        comm=comm,
    )

    # Add the extra analysis functions we want
    survey.add_analysis_func(
        get_colors_and_lrd_flags,
        "",
        cosmo=cosmo,
        nthreads=nthreads,
    )
    for frac in [0.2, 0.5, 0.8]:
        frac_key = f"{frac}".replace(".", "p")
        survey.add_analysis_func(
            lambda gal, frac=frac: gal.stars.get_attr_radius(
                "current_masses",
                frac=frac,
            ),
            f"Stars/MassRadii/{frac_key}",
        )
        survey.add_analysis_func(
            lambda gal, frac=frac: gal.gas.get_attr_radius(
                "masses",
                frac=frac,
            ),
            f"Gas/MassRadii/{frac_key}",
        )
        survey.add_analysis_func(
            lambda gal, frac=frac: gal.gas.get_attr_radius(
                "dust_masses",
                frac=frac,
            ),
            f"Gas/DustMassRadii/{frac_key}",
        )
    survey.add_analysis_func(get_stars_1d_velocity_dispersion, "Stars/VelDisp1d")
    survey.add_analysis_func(get_gas_1d_velocity_dispersion, "Gas/VelDisp1d")
    survey.add_analysis_func(get_stars_3d_velocity_dispersion, "Stars/VelDisp3d")
    survey.add_analysis_func(get_gas_3d_velocity_dispersion, "Gas/VelDisp3d")
    survey.add_analysis_func(lambda gal: gal.region, "Region")
    survey.add_analysis_func(lambda gal: gal.grp_id, "GroupID")
    survey.add_analysis_func(lambda gal: gal.subgrp_id, "SubGroupID")
    survey.add_analysis_func(lambda gal: gal.weight, "RegionWeight")
    survey.add_analysis_func(lambda gal: gal.master_index, "MasterRegionIndex")

    # Partition and load the galaxies
    indices = partition_galaxies(galaxy_weights=gal_weights)
    galaxies = load_galaxies(
        master_file_path=path,
        snap=snap,
        indices=indices,
        nthreads=nthreads,
    )

    # Add them to the survey
    survey.add_galaxies(galaxies)

    # Get the LOS optical depths
    survey.get_los_optical_depths(kernel=kernel_data)

    # Run the survey
    survey.get_spectra(cosmo=cosmo)
    survey.get_photometry_luminosities()
    survey.get_photometry_fluxes()
    survey.get_images_luminosity(fov=61 * kpc, kernel=kernel_data)
    survey.apply_psfs_luminosity()
    survey.get_images_flux(fov=61 * kpc, kernel=kernel_data)
    survey.apply_psfs_flux()

    # Save the survey
    survey.write(outpath)
