"""A script to derive synthetic observations for FLARES."""

import argparse
import time
import os
import multiprocessing as mp
import numpy as np
import h5py
from unyt import Gyr, Mpc, Msun, arcsecond, angstrom
from astropy.cosmology import Planck15 as cosmo
from mpi4py import MPI as mpi
from utils import FILTER_CODES

from synthesizer.particle import Stars, Gas
from synthesizer.particle import Galaxy
from synthesizer.filters import FilterCollection
from synthesizer import Grid
from synthesizer.kernel_functions import Kernel
from synthesizer._version import __version__

from stellar_emission_model import FLARESLRDsEmission

# Silence warnings (only because we now what we're doing)
import warnings

warnings.filterwarnings("ignore")


def _print(*args, **kwargs):
    """Overload print with rank info."""
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"[{str(rank).zfill(len(str(size)) + 1)}]: ", end="")
    print(*args, **kwargs)


def _get_galaxy(gal_ind, master_file_path, reg, snap, z):
    """
    Get a galaxy from the master file.

    Args:
        gal_ind (int): The index of the galaxy to get.
        master_file_path (str): The path to the master file.
        reg (str): The region to use.
        snap (str): The snapshot to use.
        z (float): The redshift of the snapshot.
    """
    # Get the galaxy data we need from the master file
    with h5py.File(master_file_path, "r") as hdf:
        reg_grp = hdf[reg]
        snap_grp = reg_grp[snap]
        gal_grp = snap_grp["Galaxy"]
        part_grp = snap_grp["Particle"]

        # Get the group and subgrp ids
        group_id = gal_grp["GroupNumber"][gal_ind]
        subgrp_id = gal_grp["SubGroupNumber"][gal_ind]

        # Get this galaxy's beginning and ending indices for stars
        s_len = gal_grp["S_Length"][...]
        start = np.sum(s_len[:gal_ind])
        end = np.sum(s_len[: gal_ind + 1])

        # Get this galaxy's beginning and ending indices for gas
        g_len = gal_grp["G_Length"][...]
        start_gas = np.sum(g_len[:gal_ind])
        end_gas = np.sum(g_len[: gal_ind + 1])

        # Get the star data
        star_pos = part_grp["S_Coordinates"][:, start:end].T / (1 + z) * Mpc
        star_mass = part_grp["S_Mass"][start:end] * Msun * 10**10
        star_init_mass = part_grp["S_MassInitial"][start:end] * Msun * 10**10
        star_age = part_grp["S_Age"][start:end] * Gyr
        star_met = part_grp["S_Z_smooth"][start:end]
        star_sml = part_grp["S_sml"][start:end] * Mpc

        # Get the gas data
        gas_pos = (
            part_grp["G_Coordinates"][:, start_gas:end_gas].T / (1 + z) * Mpc
        )
        gas_mass = part_grp["G_Mass"][start_gas:end_gas] * Msun * 10**10
        gas_met = part_grp["G_Z_smooth"][start_gas:end_gas]
        gas_sml = part_grp["G_sml"][start_gas:end_gas] * Mpc

        # Get the centre of potential
        centre = gal_grp["COP"][:].T[gal_ind, :] / (1 + z) * Mpc

        # Compute the angular radii of each star in arcseconds
        radii = (np.linalg.norm(star_pos - centre, axis=1)).to("kpc")
        star_ang_rad = (
            radii.value * cosmo.arcsec_per_kpc_proper(z).value * arcsecond
        )

    # Early exist if there are fewer than 100 baryons
    if star_mass.size < 100:
        return None

    # Plot a histogram of angular galaxies
    if gal_ind == 0:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(star_ang_rad.value, bins=100, histtype="step")
        ax.set_xlabel("Angular Radius (arcseconds)")
        ax.set_ylabel("Number of Stars")
        fig.savefig(f"angular_radius_histogram_{reg}_{snap}.png", dpi=300)
        plt.close(fig)

    gal = Galaxy(
        name=f"{reg}_{snap}_{gal_ind}_{group_id}_{subgrp_id}",
        redshift=z,
        stars=Stars(
            initial_masses=star_init_mass,
            current_masses=star_mass,
            ages=star_age,
            metallicities=star_met,
            redshift=z,
            coordinates=star_pos,
            smoothing_lengths=star_sml,
            centre=centre,
            angular_radii=star_ang_rad,
            radii=radii.value,
        ),
        gas=Gas(
            masses=gas_mass,
            metallicities=gas_met,
            redshift=z,
            coordinates=gas_pos,
            smoothing_lengths=gas_sml,
            centre=centre,
        ),
    )

    # Attach the extra tau_v we need for nebular attenuation around young
    # stars
    gal.stars.young_tau_v = star_met / 0.01

    # Calculate the DTM, we'll need it later
    gal.dust_to_metal_vijayan19()

    return gal


def get_flares_galaxies(
    master_file_path,
    region,
    snap,
    nthreads,
    comm,
    rank,
    size,
):
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
        s_lens = gal_grp["S_Length"][:]
        n_gals = len(s_lens)

    # Early exist if there are no galaxies
    if n_gals == 0:
        return []

    # Randomize the order of galaxies
    np.random.seed(42)
    order = np.random.permutation(n_gals)

    # Distribute galaxies by number of particles
    parts_per_rank = np.zeros(size, dtype=int)
    gals_on_rank = {rank: [] for rank in range(size)}
    for i in order:
        if s_lens[i] < 100:
            continue
        select = np.argmin(parts_per_rank)
        gals_on_rank[select].append(i)
        parts_per_rank[select] += s_lens[i]

    # Prepare the arguments for each galaxy on this rank
    args = [
        (gal_ind, master_file_path, reg, snap, z)
        for gal_ind in gals_on_rank[rank]
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
        # Create the FilterCollection
        filters = FilterCollection(filter_codes=FILTER_CODES)

        # Write the filter collection
        filters.write_filters(path=filepath)

    return filters


def get_grid(grid_name, grid_dir, filters):
    """Get a Synthesizer Grid."""
    return Grid(
        grid_name,
        grid_dir,
        # filters=filters,
        lam_lims=(1000 * angstrom, 10**6 * angstrom),
    )


def get_emission_model(grid, fesc=0.0, fesc_ly_alpha=1.0):
    """Get a StellarEmissionModel."""
    model = FLARESLRDsEmission(grid, fesc=fesc, fesc_ly_alpha=fesc_ly_alpha)

    return model


def get_kernel():
    """Get a Kernel."""
    return Kernel()


def analyse_galaxy(gal, emission_model, kern, nthreads, filters, cosmo):
    """
    Analyse a galaxy.

    This will generate all spectra and photometry for a galaxy.

    Args:
        gal (Galaxy): The galaxy to analyse.
        emission_model (StellarEmissionModel): The emission model to use.
        kern (Kernel): The kernel to use.
        filters (FilterCollection): The filter collection to use.
        cosmo (astropy.cosmology): The cosmology to use.
    """
    # Get the los tau_v
    if gal.stars.tau_v is None:
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

    # Get the integrated spectra
    gal.integrate_particle_spectra()

    # Get the observed spectra
    gal.get_observed_spectra(cosmo)

    # Get the photometry
    gal.get_photo_fluxes(filters, verbose=False)

    # Compute the half-light radius on each filter
    gal.stars.half_light_radii = {}
    for spec in gal.stars.particle_photo_fluxes.keys():
        gal.stars.half_light_radii[spec] = {}
        for filt in filters.filter_codes:
            # Sort radii
            rs = gal.stars.radii
            sinds = np.argsort(rs)
            rs = rs[sinds]

            # Get the cumalitive flux and half the total
            cum_fluxes = np.cumsum(
                gal.stars.particle_photo_fluxes[spec][filt][sinds].value
            )
            half_flux = 0.5 * cum_fluxes[-1]

            # Get the half light radius (using the closest particle to
            # the half flux)
            gal.stars.half_light_radii[spec][filt] = rs[
                np.argmin(np.abs(cum_fluxes - half_flux))
            ]

    return gal


def get_image():
    pass


def write_results(galaxies, path, grid_name, filters, comm, rank, size):
    """Write the results to a file."""
    # Loop over galaxies and unpacking all the data we'll write out
    fluxes = {}
    fnus = {}
    compactnesses = {}
    uv_slopes = {}
    ir_slopes = {}
    sizes = {}
    group_ids = []
    subgroup_ids = []
    indices = []
    for gal in galaxies:
        # Get the group and subgroup ids
        indices.append(int(gal.name.split("_")[3]))
        group_ids.append(int(gal.name.split("_")[4]))
        subgroup_ids.append(int(gal.name.split("_")[5]))

        # Get the integrated observed spectra
        for key, spec in gal.stars.spectra.items():
            fnus.setdefault(key, []).append(spec._fnu)

        # Get the photometry
        for key, photcol in gal.stars.photo_fluxes.items():
            fluxes.setdefault(key, {})
            for filt, phot in photcol.items():
                fluxes[key].setdefault(filt, []).append(phot)

        # Get the compactness
        for key in ["reprocessed", "attenuated"]:
            compactnesses.setdefault(key, {})
            for filt in filters.filter_codes:
                compactnesses[key].setdefault(filt, []).append(
                    gal.stars.photo_fluxes[f"0p4_aperture_{key}"][filt].value
                    / gal.stars.photo_fluxes[f"0p2_aperture_{key}"][filt].value
                )

        # Get slopes
        for key, spectra in gal.stars.spectra.items():
            uv_slopes.setdefault(key, []).append(
                spectra.measure_beta(window=(1500 * angstrom, 3000 * angstrom))
            )
            ir_slopes.setdefault(key, []).append(
                spectra.measure_beta(window=(4400 * angstrom, 7500 * angstrom))
            )

        # Get the sizes
        for spec in gal.stars.half_light_radii.keys():
            for filt in gal.stars.half_light_radii[spec].keys():
                sizes.setdefault(spec, {}).setdefault(filt, []).append(
                    gal.stars.half_light_radii[spec][filt]
                )

    # Collect output data onto rank 0
    fnu_per_rank = comm.gather(fnus, root=0)
    flux_per_rank = comm.gather(fluxes, root=0)
    comp_per_rank = comm.gather(compactnesses, root=0)
    group_per_rank = comm.gather(group_ids, root=0)
    subgroup_per_rank = comm.gather(subgroup_ids, root=0)
    index_per_rank = comm.gather(indices, root=0)
    uv_slope_per_rank = comm.gather(uv_slopes, root=0)
    ir_slope_per_rank = comm.gather(ir_slopes, root=0)
    size_per_rank = comm.gather(sizes, root=0)

    # Early exit if we're not rank 0
    if rank != 0:
        return

    # Concatenate the data
    fnus = {}
    fluxes = {}
    compactnesses = {}
    group_ids = []
    subgroup_ids = []
    indices = []
    uv_slopes = {}
    ir_slopes = {}
    sizes = {}
    for (
        fnu,
        flux,
        comp,
        group,
        subgroup,
        index,
        uv_slope,
        ir_slope,
        size,
    ) in zip(
        fnu_per_rank,
        flux_per_rank,
        comp_per_rank,
        group_per_rank,
        subgroup_per_rank,
        index_per_rank,
        uv_slope_per_rank,
        ir_slope_per_rank,
        size_per_rank,
    ):
        for key, spec in fnu.items():
            fnus.setdefault(key, []).extend(spec)
        for key, phot in flux.items():
            fluxes.setdefault(key, {})
            for filt, phot_arr in phot.items():
                fluxes[key].setdefault(filt, []).extend(phot_arr)
        for key, comps in comp.items():
            compactnesses.setdefault(key, {})
            for filt in filters.filter_codes:
                compactnesses[key].setdefault(filt, []).extend(comps[filt])
        for key, slopes in uv_slope.items():
            uv_slopes.setdefault(key, []).extend(slopes)
        for key, slopes in ir_slope.items():
            ir_slopes.setdefault(key, []).extend(slopes)
        for key, d in size.items():
            sizes.setdefault(key, {})
            for filt, size_arr in d.items():
                sizes[key].setdefault(filt, []).extend(size_arr)
        group_ids.extend(group)
        subgroup_ids.extend(subgroup)
        indices.extend(index)

    # Get the units for each dataset
    units = {"fnu": "erg/s/cm**2/Hz", "flux": "erg/s/cm^**2", "hlr": "kpc"}

    # Sort the data by galaxy index
    sort_indices = np.argsort(indices)
    for key, spec in fnus.items():
        fnus[key] = [spec[i] for i in sort_indices]
    for key, phot in fluxes.items():
        fluxes[key] = {
            filt: [phot[filt][i] for i in sort_indices] for filt in phot
        }
    for key, comp in compactnesses.items():
        compactnesses[key] = {
            filt: [comp[filt][i] for i in sort_indices] for filt in comp
        }
    for key, slopes in uv_slopes.items():
        uv_slopes[key] = [slopes[i] for i in sort_indices]
    for key, slopes in ir_slopes.items():
        ir_slopes[key] = [slopes[i] for i in sort_indices]
    for key, d in sizes.items():
        sizes[key] = {filt: [d[filt][i] for i in sort_indices] for filt in d}
    group_ids = [group_ids[i] for i in sort_indices]
    subgroup_ids = [subgroup_ids[i] for i in sort_indices]
    indices = [indices[i] for i in sort_indices]

    # Write output out to file
    with h5py.File(path, "w") as hdf:
        # Write the group and subgroup ids
        hdf.create_dataset("GroupNumber", data=group_ids)
        hdf.create_dataset("SubGroupNumber", data=subgroup_ids)

        # Store the grid used and the Synthesizer version
        hdf.attrs["Grid"] = grid_name
        hdf.attrs["SynthesizerVersion"] = __version__

        # Create groups for the data
        fnu_grp = hdf.create_group("ObservedSpectra")
        flux_grp = hdf.create_group("ObservedPhotometry")
        comp_grp = hdf.create_group("Compactness")

        # Write the integrated observed spectra
        for key, fnu in fnus.items():
            dset = fnu_grp.create_dataset(
                key,
                data=np.array(fnu),
            )
            dset.attrs["Units"] = units["fnu"]

        # Write the photometry
        for key, flux in fluxes.items():
            filt_grp = flux_grp.create_group(key)
            for filt, phot in flux.items():
                dset = filt_grp.create_dataset(
                    filt,
                    data=np.array(phot),
                )
                dset.attrs["Units"] = units["flux"]

        # Write the compactness
        for key, comp in compactnesses.items():
            filt_grp = comp_grp.create_group(key)
            for filt, comp_arr in comp.items():
                dset = filt_grp.create_dataset(
                    filt,
                    data=np.array(comp_arr),
                )
                dset.attrs["Units"] = "dimensionless"

        # Write the UV slopes
        uv_grp = hdf.create_group("UVSlopes")
        for key, slopes in uv_slopes.items():
            dset = uv_grp.create_dataset(
                key,
                data=np.array(slopes),
            )
            dset.attrs["Units"] = "dimensionless"

        # Write the IR slopes
        ir_grp = hdf.create_group("OpticalSlopes")
        for key, slopes in ir_slopes.items():
            dset = ir_grp.create_dataset(
                key,
                data=np.array(slopes),
            )
            dset.attrs["Units"] = "dimensionless"

        # Write the half light radii
        hlr_grp = hdf.create_group("HalfLightRadii")
        for key, d in sizes.items():
            filt_grp = hlr_grp.create_group(key)
            for filt, size_arr in d.items():
                dset = filt_grp.create_dataset(
                    filt,
                    data=np.array(size_arr),
                )
                dset.attrs["Units"] = units["hlr"]


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

    # Get MPI info
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse the arguments
    args = parser.parse_args()
    path = args.master_file_path
    grid_name = args.grid
    grid_dir = args.grid_dir
    region = args.region
    snap = snapshots[args.snap]
    nthreads = args.nthreads

    # Define the output path
    outpath = f"data/pure_stellar_{str(region).zfill(2)}_{snap}.hdf5"

    # If the output already exists just exit
    if os.path.exists(outpath):
        _print(f"Output file {outpath} already exists.")
        exit(0)

    start = time.time()

    # Get the filters
    filt_start = time.time()
    filters = get_flares_filters("lrd_filters.hdf5")
    filt_end = time.time()
    _print(f"Getting filters took {filt_end - filt_start:.2f} seconds.")

    # Get the grid
    grid_start = time.time()
    grid = get_grid(grid_name, grid_dir, filters)
    grid_end = time.time()

    # Get the emission model
    start_emission = time.time()
    emission_model = get_emission_model(grid)
    end_emission = time.time()
    _print(
        f"Getting the emission model took "
        f"{end_emission - start_emission:.2f} seconds."
    )

    # If we're on rank 0 and the first region and snapshot plot the model
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0 and region == 0 and snap == snapshots[0]:
        fig, ax = emission_model.plot_emission_tree(fontsize=8)
        fig.savefig("emission_tree.png", dpi=300, bbox_inches="tight")

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
        comm,
        rank,
        size,
    )
    read_end = time.time()
    _print(
        f"Creating {len(galaxies)} galaxies took "
        f"{read_end - read_start:.2f} seconds."
    )

    # Analyse the galaxy
    gal_start = time.time()
    galaxies = [
        analyse_galaxy(
            gal,
            emission_model,
            kern,
            nthreads,
            filters,
            cosmo,
        )
        for gal in galaxies
    ]
    gal_end = time.time()
    _print(
        f"Analysing {len(galaxies)} galaxies took "
        f"{gal_end - gal_start:.2f} seconds."
    )

    # Write out the results
    write_start = time.time()
    write_results(
        galaxies,
        outpath,
        grid_name,
        filters,
        comm,
        rank,
        size,
    )

    comm.barrier()

    end = time.time()
    _print(f"Total time: {end - start:.2f} seconds.")
