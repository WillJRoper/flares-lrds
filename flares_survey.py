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
from synthesizer.grid import Grid
from synthesizer.instruments import InstrumentCollection
from synthesizer.kernel_functions import Kernel
from synthesizer.particle import BlackHoles, Galaxy, Gas, Stars
from synthesizer.pipeline import Pipeline
from unyt import Gyr, Mpc, Msun, angstrom, arcsecond, km, kpc, s, yr

from combined_emission_model import FLARESLOSCombinedEmission
from extra_analysis_funcs import (
    get_bh_average_accretion_rate,
    get_bh_central_accretion_rate,
    get_bh_central_mass,
    get_bh_number,
    get_bh_total_mass,
    get_colors_and_lrd_flags,
    get_gas_1d_velocity_dispersion,
    get_gas_3d_velocity_dispersion,
    get_optical_depth,
    get_pixel_based_hlr,
    get_stars_1d_velocity_dispersion,
    get_stars_3d_velocity_dispersion,
)
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
    if np.sum(mask) < 100:
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
    nranks = mpi.COMM_WORLD.Get_size()
    this_rank = mpi.COMM_WORLD.Get_rank()

    # Create an array of galaxy indices
    gal_inds = np.arange(len(galaxy_weights))

    # Sanitise away galaxies below the threshold
    gal_inds = gal_inds[galaxy_weights >= 100]

    # Split the galaxies between the processes
    indices = np.array_split(gal_inds, nranks)

    return indices[this_rank]


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
        grid,
        agn_template_file,
    )

    # Limit the spectra to be saved
    model.save_spectra(*save_spectra)

    return model


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

    # Get the SPH kernel
    kernel_data = Kernel().get_kernel()

    # Set up the pipeline
    pipeline = Pipeline(
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
    pipeline.add_analysis_func(
        get_colors_and_lrd_flags,
        "",
        cosmo=cosmo,
        nthreads=nthreads,
    )
    for frac in [0.2, 0.5, 0.8]:
        frac_key = f"{frac}".replace(".", "p")
        pipeline.add_analysis_func(
            lambda gal, frac=frac: gal.stars.get_attr_radius(
                "current_masses",
                frac=frac,
            ),
            f"Stars/MassRadii/{frac_key}",
        )
        pipeline.add_analysis_func(
            lambda gal, frac=frac: gal.gas.get_attr_radius(
                "masses",
                frac=frac,
            ),
            f"Gas/MassRadii/{frac_key}",
        )
        pipeline.add_analysis_func(
            lambda gal, frac=frac: gal.gas.get_attr_radius(
                "dust_masses",
                frac=frac,
            ),
            f"Gas/DustMassRadii/{frac_key}",
        )
    pipeline.add_analysis_func(get_pixel_based_hlr, "HalfLightRadii")
    pipeline.add_analysis_func(
        lambda gal: get_pixel_based_hlr(gal.stars),
        "Stars/HalfLightRadii",
    )
    pipeline.add_analysis_func(get_stars_1d_velocity_dispersion, "Stars/VelDisp1d")
    pipeline.add_analysis_func(get_gas_1d_velocity_dispersion, "Gas/VelDisp1d")
    pipeline.add_analysis_func(get_stars_3d_velocity_dispersion, "Stars/VelDisp3d")
    pipeline.add_analysis_func(get_gas_3d_velocity_dispersion, "Gas/VelDisp3d")
    pipeline.add_analysis_func(lambda gal: gal.region, "Region")
    pipeline.add_analysis_func(lambda gal: gal.grp_id, "GroupID")
    pipeline.add_analysis_func(lambda gal: gal.subgrp_id, "SubGroupID")
    pipeline.add_analysis_func(lambda gal: gal.weight, "RegionWeight")
    pipeline.add_analysis_func(lambda gal: gal.master_index, "MasterRegionIndex")
    pipeline.add_analysis_func(lambda gal: gal.redshift, "Redshift")
    pipeline.add_analysis_func(get_bh_number, "BlackHoles/NBlackHoles")
    pipeline.add_analysis_func(get_bh_total_mass, "BlackHoles/TotalMass")
    pipeline.add_analysis_func(get_bh_central_mass, "BlackHoles/CentralMass")
    pipeline.add_analysis_func(
        get_bh_average_accretion_rate,
        "BlackHoles/AverageAccretionRate",
    )
    pipeline.add_analysis_func(
        get_bh_central_accretion_rate,
        "BlackHoles/CentralAccretionRate",
    )
    pipeline.add_analysis_func(
        lambda gal: get_optical_depth(gal.stars),
        "Stars/VBandOpticalDepth",
    )
    pipeline.add_analysis_func(
        lambda gal: get_optical_depth(gal.black_holes),
        "BlackHoles/VBandOpticalDepth",
    )
    # pipeline.add_analysis_func(get_UV_slopes, "UVSlope")
    # pipeline.add_analysis_func(get_IR_slopes, "IRSlope")
    # pipeline.add_analysis_func(
    #     lambda gal: get_UV_slopes(gal.stars),
    #     "Stars/UVSlope",
    # )
    # pipeline.add_analysis_func(
    #     lambda gal: get_IR_slopes(gal.stars),
    #     "Stars/IRSlope",
    # )
    # pipeline.add_analysis_func(
    #     lambda gal: get_UV_slopes(gal.black_holes), "BlackHoles/UVSlope"
    # )
    # pipeline.add_analysis_func(
    #     lambda gal: get_IR_slopes(gal.black_holes), "BlackHoles/IRSlope"
    # )

    # Partition and load the galaxies
    indices = partition_galaxies(galaxy_weights=gal_weights)
    galaxies = load_galaxies(
        master_file_path=path,
        snap=snap,
        indices=indices,
        nthreads=nthreads,
    )

    # Add them to the pipeline
    pipeline.add_galaxies(galaxies)

    # Get the LOS optical depths
    pipeline.get_los_optical_depths(kernel=kernel_data)

    # Run the pipeline
    pipeline.get_spectra(cosmo=cosmo)
    pipeline.get_photometry_luminosities()
    pipeline.get_photometry_fluxes()

    # No longer need the particle spectra
    for gal in pipeline.galaxies:
        gal.clear_all_spectra()

    # pipeline.get_images_luminosity(fov=61 * kpc, kernel=kernel_data)
    # pipeline.apply_psfs_luminosity()
    pipeline.get_images_flux(fov=61 * kpc, kernel=kernel_data)
    pipeline.apply_psfs_flux()

    # Save the pipeline
    pipeline.write(
        outpath,
        output_lnu=False,
        output_fnu=False,
        output_images_fnu=False,
        output_images_fnu_psf=False,
    )
    pipeline.combine_files_virtual()
