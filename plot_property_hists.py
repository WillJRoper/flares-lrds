"""A script for plotting histograms of galaxy properties."""
import argparse
import numpy as np

from utils import (
    get_master_data,
    plot_step_hist,
    get_synth_data,
)

# Define the parser
parser = argparse.ArgumentParser(
    description="Plot the compactness of galaxies."
)
parser.add_argument(
    "--type",
    type=str,
    default="stellar",
    help="The type of data to plot.",
)
parser.add_argument(
    "--spectra",
    type=str,
    nargs="+",
    default=[
        "attenuated",
    ],
    help="The spectra to plot.",
)
parser.add_argument(
    "--master",
    type=str,
    required=True,
    help="The master file to use.",
)
parser.add_argument(
    "--normalise",
    action="store_true",
    help="Normalise the histograms.",
)

# Parse the arguments
args = parser.parse_args()

# Define the data file
if args.type == "stellar":
    data_file = "data/pure_stellar_<region>_<snap>.hdf5"
elif args.type == "agn":
    data_file = "data/pure_agn_<region>_<snap>.hdf5"
else:
    data_file = "data/combined_<region>_<snap>.hdf5"

# Define plotting parameters
bins = 30

# Loop over the spectra
for spectra in args.spectra:
    # Get the synthesizer data
    (
        fluxes,
        colors,
        red1,
        red2,
        sizes,
        masks,
        indices,
        weights,
    ) = get_synth_data(data_file, spectra, get_weights=True)

    # Get the master file data excluding the masks
    all_dtms = get_master_data(args.master, indices, "DTM")
    all_m200 = get_master_data(args.master, indices, "M200")
    all_r200 = get_master_data(args.master, indices, "R200")
    all_star_metals = get_master_data(
        args.master,
        indices,
        "Metallicity/CurrentMassWeightedStellarZ",
    )
    all_gas_metals = get_master_data(
        args.master,
        indices,
        "Metallicity/MassWeightedGasZ",
    )
    all_star_masses = get_master_data(
        args.master,
        indices,
        "Mstar_aperture/30",
    )
    all_gas_masses = get_master_data(
        args.master,
        indices,
        "Mgas_aperture/30",
    )
    all_bh_masses = get_master_data(
        args.master,
        indices,
        "Mbh",
    )
    all_sfr = get_master_data(
        args.master,
        indices,
        "SFR_aperture/30/100Myr",
    )
    all_sfr_10 = get_master_data(
        args.master,
        indices,
        "SFR_aperture/30/10Myr",
    )
    all_star_ages = get_master_data(
        args.master,
        indices,
        "StellarAges/MassWeightedStellarAge",
    )

    # Loop over snapshots
    for snap in all_dtms.keys():
        # Early exist if there are no galaxies in the mask
        if np.sum(masks[snap]) == 0:
            continue

        # Plot the histogram
        plot_step_hist(
            f"{spectra}/{snap}/dust_to_metal_{snap}_hist",
            label=r"$\mathrm{DTM}$",
            bins=bins,
            normalise=args.normalise,
            weights={
                "All_Galaxies": weights[snap],
                "LRDs": weights[snap][masks[snap]],
            },
            All_Galaxies=all_dtms[snap],
            LRDs=all_dtms[snap][masks[snap]],
        )
        plot_step_hist(
            f"{spectra}/{snap}/m200_{snap}_hist",
            label=r"$\log_{10}(M_{200} / [M_\odot])$",
            bins=bins,
            normalise=args.normalise,
            weights={
                "All_Galaxies": weights[snap],
                "LRDs": weights[snap][masks[snap]],
            },
            All_Galaxies=(all_m200[snap] * 10**10),
            LRDs=(all_m200[snap][masks[snap]] * 10**10),
        )
        plot_step_hist(
            f"{spectra}/{snap}/r200_{snap}_hist",
            label=r"$R_{200} / [\mathrm{cMpc}]$",
            bins=bins,
            normalise=args.normalise,
            weights={
                "All_Galaxies": weights[snap],
                "LRDs": weights[snap][masks[snap]],
            },
            log=False,
            All_Galaxies=all_r200[snap],
            LRDs=all_r200[snap][masks[snap]],
        )
        plot_step_hist(
            f"{spectra}/{snap}/star_metals_{snap}_hist",
            label=r"$\log_{10}(Z_{\star})$",
            bins=bins,
            normalise=args.normalise,
            weights={
                "All_Galaxies": weights[snap],
                "LRDs": weights[snap][masks[snap]],
            },
            All_Galaxies=(all_star_metals[snap]),
            LRDs=(all_star_metals[snap][masks[snap]]),
        )
        plot_step_hist(
            f"{spectra}/{snap}/gas_metals_{snap}_hist",
            label=r"$\log_{10}(Z_{\mathrm{gas}})$",
            bins=bins,
            normalise=args.normalise,
            weights={
                "All_Galaxies": weights[snap],
                "LRDs": weights[snap][masks[snap]],
            },
            All_Galaxies=(all_gas_metals[snap]),
            LRDs=(all_gas_metals[snap][masks[snap]]),
        )
        plot_step_hist(
            f"{spectra}/{snap}/star_masses_{snap}_hist",
            label=r"$\log_{10}(M_{\star} / [M_\odot])$",
            bins=bins,
            normalise=args.normalise,
            weights={
                "All_Galaxies": weights[snap],
                "LRDs": weights[snap][masks[snap]],
            },
            All_Galaxies=(all_star_masses[snap] * 10**10),
            LRDs=(all_star_masses[snap][masks[snap]] * 10**10),
        )
        plot_step_hist(
            f"{spectra}/{snap}/gas_masses_{snap}_hist",
            label=r"$\log_{10}(M_{\mathrm{gas}} / [M_\odot])$",
            bins=bins,
            normalise=args.normalise,
            weights={
                "All_Galaxies": weights[snap],
                "LRDs": weights[snap][masks[snap]],
            },
            All_Galaxies=(all_gas_masses[snap] * 10**10),
            LRDs=(all_gas_masses[snap][masks[snap]] * 10**10),
        )
        plot_step_hist(
            f"{spectra}/{snap}/bh_masses_{snap}_hist",
            label=r"$\log_{10}(M_{\mathrm{BH}} / [M_\odot])$",
            bins=bins,
            normalise=args.normalise,
            weights={
                "All_Galaxies": weights[snap],
                "LRDs": weights[snap][masks[snap]],
            },
            All_Galaxies=(all_bh_masses[snap] * 10**10),
            LRDs=(all_bh_masses[snap][masks[snap]] * 10**10),
        )
        plot_step_hist(
            f"{spectra}/{snap}/sfr_{snap}_hist",
            label=r"$\log_{10}(\mathrm{SFR}_{100} / [M_\odot / \mathrm{yr}])$",
            bins=bins,
            normalise=args.normalise,
            weights={
                "All_Galaxies": weights[snap],
                "LRDs": weights[snap][masks[snap]],
            },
            All_Galaxies=(all_sfr[snap]),
            LRDs=(all_sfr[snap][masks[snap]]),
        )
        plot_step_hist(
            f"{spectra}/{snap}/star_ages_{snap}_hist",
            label=r"$\mathrm{Age}_{\star} / [Gyr]$",
            bins=bins,
            normalise=args.normalise,
            weights={
                "All_Galaxies": weights[snap],
                "LRDs": weights[snap][masks[snap]],
            },
            log=False,
            All_Galaxies=all_star_ages[snap],
            LRDs=all_star_ages[snap][masks[snap]],
        )
        plot_step_hist(
            f"{spectra}/{snap}/ssfr_{snap}_hist",
            label=r"$\log_{10}(\mathrm{sSFR}_{100} /$ [yr])",
            bins=bins,
            normalise=args.normalise,
            weights={
                "All_Galaxies": weights[snap],
                "LRDs": weights[snap][masks[snap]],
            },
            All_Galaxies=(all_sfr[snap] / (all_star_masses[snap] * 10**10)),
            LRDs=(
                all_sfr[snap][masks[snap]]
                / (all_star_masses[snap][masks[snap]] * 10**10)
            ),
        )
        plot_step_hist(
            f"{spectra}/{snap}/sfr_10_{snap}_hist",
            label=r"$\log_{10}(\mathrm{SFR}_{10} / [M_\odot / \mathrm{yr}])$",
            bins=bins,
            normalise=args.normalise,
            weights={
                "All_Galaxies": weights[snap],
                "LRDs": weights[snap][masks[snap]],
            },
            All_Galaxies=(all_sfr_10[snap]),
            LRDs=(all_sfr_10[snap][masks[snap]]),
        )
        plot_step_hist(
            f"{spectra}/{snap}/ssfr_10_{snap}_hist",
            label=r"$\log_{10}(\mathrm{sSFR}_{10} /$ [yr])",
            bins=bins,
            normalise=args.normalise,
            weights={
                "All_Galaxies": weights[snap],
                "LRDs": weights[snap][masks[snap]],
            },
            All_Galaxies=(
                all_sfr_10[snap] / (all_star_masses[snap] * 10**10)
            ),
            LRDs=(
                all_sfr_10[snap][masks[snap]]
                / (all_star_masses[snap][masks[snap]] * 10**10)
            ),
        )
