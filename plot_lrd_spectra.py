"""A script for LRD spectra."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from unyt import erg, s, cm, Hz, nJy, arcsecond
from astropy.cosmology import Planck15 as cosmo

from synthesizer.conversions import angular_to_spatial_at_z
from synthesizer.sed import Sed, plot_observed_spectra

from utils import (
    get_synth_data_with_imgs,
    savefig,
    get_galaxy_identifiers,
    get_synth_spectra,
)
from synthesize_flares_with_agn import get_grid

# Define the parser
parser = argparse.ArgumentParser(description="Plot the SFZHs of galaxies.")
parser.add_argument(
    "--type",
    type=str,
    default="stellar",
    help="The type of data to plot.",
)
parser.add_argument(
    "--grid",
    type=str,
    help="The file name of the Synthesizer gird.",
)
parser.add_argument(
    "--master",
    type=str,
    required=True,
    help="The master file to use.",
)
parser.add_argument(
    "--grid-dir",
    type=str,
    help="The directory of the Synthesizer grid.",
)

# Parse the arguments
args = parser.parse_args()

# Parse the arguments
args = parser.parse_args()

# Define the data file
data_file = "data/combined_<region>_<snap>.hdf5"

# Get the grid
grid = get_grid(args.grid, args.grid_dir, None)

# Get the spectra arrays
stellar_att_spectra = get_synth_spectra(
    data_file, "attenuated", cut_on="attenuated"
)
agn_att_spectra = get_synth_spectra(
    data_file, "agn_attenuated", cut_on="attenuated"
)
stellar_rep_spectra = get_synth_spectra(
    data_file, "reprocessed", cut_on="attenuated"
)
agn_rep_spectra = get_synth_spectra(
    data_file, "agn_intrinsic", cut_on="attenuated"
)

# Get the synthesizer data
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

# Get the galaxy ids for labelling
gal_ids = get_galaxy_identifiers(args.master, indices)

# Apply the masks
for snap in stellar_att_spectra:
    stellar_att_spectra[snap] = stellar_att_spectra[snap][masks[snap]]
    agn_att_spectra[snap] = agn_att_spectra[snap][masks[snap]]
    stellar_rep_spectra[snap] = stellar_rep_spectra[snap][masks[snap]]
    agn_rep_spectra[snap] = agn_rep_spectra[snap][masks[snap]]
    gal_ids[snap] = gal_ids[snap][masks[snap]]

# Loop over snapshots
for snap in stellar_att_spectra:
    # Skip if there are no galaxies
    if len(stellar_att_spectra[snap]) == 0:
        continue

    # Get redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Get the seds
    seds = []
    for i in range(len(stellar_att_spectra)):
        seds.append({})
        seds[-1]["stellar_attenuated"] = Sed(
            grid.lam,
        )
        seds[-1]["stellar_attenuated"].fnu = (
            stellar_att_spectra[snap][i, :] * erg / s / cm**2 / Hz
        )
        seds[-1]["stellar_attenuated"].obslam = grid.lam * (1 + z)
        seds[-1]["agn_attenuated"] = Sed(
            grid.lam,
        )
        seds[-1]["agn_attenuated"].fnu = (
            agn_att_spectra[snap][i, :] * erg / s / cm**2 / Hz
        )
        seds[-1]["agn_attenuated"].obslam = grid.lam * (1 + z)
        seds[-1]["stellar_reprocessed"] = Sed(
            grid.lam,
        )
        seds[-1]["stellar_reprocessed"].fnu = (
            stellar_rep_spectra[snap][i, :] * erg / s / cm**2 / Hz
        )
        seds[-1]["stellar_reprocessed"].obslam = grid.lam * (1 + z)
        seds[-1]["agn_reprocessed"] = Sed(
            grid.lam,
        )
        seds[-1]["agn_reprocessed"].fnu = (
            agn_rep_spectra[snap][i, :] * erg / s / cm**2 / Hz
        )
        seds[-1]["agn_reprocessed"].obslam = grid.lam * (1 + z)

    # Loop over spectra plotting them
    for spec in seds:
        fig, ax = plot_observed_spectra(
            spec,
            show=False,
            redshift=z,
        )

        savefig(
            fig, f"spectra/spectra_{'_'.join(gal_ids[snap][i])}_{snap}.png"
        )
