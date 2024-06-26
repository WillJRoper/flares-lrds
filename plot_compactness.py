"""A script for plotting the compactness of galaxies."""
import argparse
import matplotlib.colors as mcolors

from utils import (
    get_masked_synth_data,
    get_synth_data,
    SNAPSHOTS,
    plot_masked_unmasked_hexbins,
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

# Parse the arguments
args = parser.parse_args()

# Define the data file
if args.type == "stellar":
    data_file = "data/pure_stellar_<region>_<snap>.hdf5"
elif args.type == "agn":
    data_file = "data/pure_agn_<region>_<snap>.hdf5"
else:
    data_file = "data/combined_<region>_<snap>.hdf5"

# Define the spectra keys we'll read
spectra_keys = ["attenuated", "reprocessed"]

# Get the fluxes, colors and masks
(
    att_fluxes,
    att_colors,
    att_red1,
    att_red2,
    att_sizes,
    att_masks,
    _,
) = get_synth_data(data_file, "attenuated")
(
    rep_fluxes,
    rep_colors,
    rep_red1,
    rep_red2,
    rep_sizes,
    rep_masks,
    _,
) = get_synth_data(data_file, "reprocessed")

# Define plotting parameters
gridsize = 50
norm = mcolors.LogNorm(1, 10**3.5, clip=True)
extent = (-1.1, 1.3, 0, 20)

# Get the data
att_hlrs = get_masked_synth_data(
    data_file,
    "HalfLightRadii/attenuated/JWST/NIRCam.F444W",
)
rep_hlrs = get_masked_synth_data(
    data_file,
    "HalfLightRadii/reprocessed/JWST/NIRCam.F444W",
)
att_lr_80s = get_masked_synth_data(
    data_file,
    "LightRadii80/attenuated/JWST/NIRCam.F444W",
)
rep_lr_80s = get_masked_synth_data(
    data_file,
    "LightRadii80/reprocessed/JWST/NIRCam.F444W",
)
att_lr_20s = get_masked_synth_data(
    data_file,
    "LightRadii20/attenuated/JWST/NIRCam.F444W",
)
rep_lr_20s = get_masked_synth_data(
    data_file,
    "LightRadii20/reprocessed/JWST/NIRCam.F444W",
)

# Plot the data
for snap in SNAPSHOTS:
    plot_masked_unmasked_hexbins(
        att_hlrs[snap],
        att_lr_80s[snap] / att_lr_20s[snap],
        att_masks[snap],
        extent,
        norm,
        "$R_{1/2}$ / [kpc]",
        "$C_{82}$",
        f"compactness_attenuated_{snap}",
        yscale="linear",
    )
    plot_masked_unmasked_hexbins(
        rep_hlrs[snap],
        rep_lr_80s[snap] / rep_lr_20s[snap],
        rep_masks[snap],
        extent,
        norm,
        "$R_{1/2}$ / [kpc]",
        "$C_{82}$",
        f"compactness_reprocessed_{snap}",
        yscale="linear",
    )
