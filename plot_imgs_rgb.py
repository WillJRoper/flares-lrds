"""A script for plotting images."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from unyt import arcsecond
from astropy.cosmology import Planck15 as cosmo

from synthesizer.conversions import angular_to_spatial_at_z

from utils import get_synth_data_with_imgs, savefig, get_galaxy_identifiers

# Define the parser
parser = argparse.ArgumentParser(description="Plot images of galaxies.")
parser.add_argument(
    "--spec-type",
    type=str,
    default="attenuated",
    help="The spectra type of data to plot.",
)
parser.add_argument(
    "--master",
    type=str,
    default="/cosma7/data/dp004/dc-payy1/my_files//flares_pipeline/data/flares.hdf5",
    help="The master data file.",
)

# Parse the arguments
args = parser.parse_args()

# Define the data file
data_file = "data/combined_<region>_<snap>.hdf5"

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
) = get_synth_data_with_imgs(data_file, args.spec_type)

# Get the galaxy ids for labelling
gal_ids = get_galaxy_identifiers(args.master, indices)

# How many filters do we have?
nfilt = len(images["010_z005p000"].keys())

# Define the RGB filters
red = [(0.5, "F444W"), (0.5, "F356W")]
green = [(0.5, "F200W"), (0.5, "F277W")]
blue = [(0.5, "F115W"), (0.5, "F150W")]

# Apply masks to the images
for snap in images:
    for filt in images[snap]:
        images[snap][filt] = images[snap][filt][masks[snap], :, :]
    gal_ids[snap] = gal_ids[snap][masks[snap]]

# Count the total number of LRDs
n_lrd = 0
for snap in images:
    n_lrd += len(images[snap]["F115W"])
print(f"Total number of LRDs: {n_lrd}")

# Loop over regions
for snap in images:
    # Are there any LRD images to plot?
    if len(images[snap]["F115W"]) == 0:
        continue
    print(f"Plotting images for {snap} ({len(images[snap]['F115W'])} images)")

    # Get redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Define the resolution
    res = 0.031 * arcsecond
    res_kpc = angular_to_spatial_at_z(res, cosmo, z)

    # Loop over images
    for i in range(len(images[snap]["F115W"])):
        # Setup the plot
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)

        # Create the RGB image
        rgb = np.zeros(
            (
                images[snap][red[0][1]].shape[1],
                images[snap][red[0][1]].shape[2],
                3,
            )
        )
        red_flux = np.sum(
            [images[snap][filt[1]][i] * filt[0] for filt in red], axis=0
        )
        green_flux = np.sum(
            [images[snap][filt[1]][i] * filt[0] for filt in green], axis=0
        )
        blue_flux = np.sum(
            [images[snap][filt[1]][i] * filt[0] for filt in blue], axis=0
        )
        rgb[:, :, 0] = red_flux
        rgb[:, :, 1] = green_flux
        rgb[:, :, 2] = blue_flux

        # Normalise the rgb image
        rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))

        # Plot the images
        im = ax.imshow(rgb)
        ax.axis("off")

        # # Draw a dashed aperture at 0.2" and 0.4" (we need to convert these
        # # to kpc and then to pixels)
        # aper1 = angular_to_spatial_at_z(0.2 * arcsecond, cosmo, z) / res_kpc
        # aperture1 = plt.Circle(
        #     (rgb.shape[1] / 2, rgb.shape[0] / 2),
        #     aper1,
        #     color="white",
        #     fill=False,
        #     linestyle="--",
        # )
        # ax.add_artist(aperture1)
        # aper2 = angular_to_spatial_at_z(0.4 * arcsecond, cosmo, z) / res_kpc
        # aperture2 = plt.Circle(
        #     (rgb.shape[1] / 2, rgb.shape[0] / 2),
        #     aper2,
        #     color="white",
        #     fill=False,
        #     linestyle="--",
        # )
        # ax.add_artist(aperture2)

        savefig(
            fig,
            f"images/{args.spec_type}/rgb_{args.spec_type}_"
            f"{snap}_{'_'.join([str(s) for s in gal_ids[snap][i]])}",
        )
