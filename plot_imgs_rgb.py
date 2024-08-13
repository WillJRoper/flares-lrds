"""A script for plotting images."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from unyt import erg, s, cm, Hz, nJy

from utils import get_synth_data_with_imgs, savefig

# Define the parser
parser = argparse.ArgumentParser(description="Plot images of galaxies.")
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

# How many filters do we have?
nfilt = len(images["010_z005p000"].keys())

# Define the ROGB filters
red = ["F444W"]
orange = ["F356W"]
green = ["F200W", "F277W"]
blue = ["F115W", "F150W"]

# Apply masks to the images
for snap in images:
    for filt in images[snap]:
        images[snap][filt] = images[snap][filt][masks[snap], :, :]

# Loop over regions
for snap in images:
    # Are there any LRD images to plot?
    if len(images[snap]["F115W"]) == 0:
        continue

    print(f"Plotting images for {snap} ({len(images[snap]['F115W'])} images)")

    # Loop over images
    for i in range(len(images[snap]["F115W"])):
        # Setup the plot
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)

        # Create the RGB image
        rgb = np.zeros(
            (images[snap][red[0]].shape[1], images[snap][red[0]].shape[2], 3)
        )
        red_flux = np.sum([images[snap][filt][i] for filt in red], axis=0)
        orange_flux = np.sum(
            [images[snap][filt][i] for filt in orange], axis=0
        )
        green_flux = np.sum([images[snap][filt][i] for filt in green], axis=0)
        blue_flux = np.sum([images[snap][filt][i] for filt in blue], axis=0)
        rgb[:, :, 0] = red_flux + 0.6 * orange_flux
        rgb[:, :, 1] = green_flux + 0.4 * orange_flux
        rgb[:, :, 2] = blue_flux

        # Plot the images
        im = ax.imshow(rgb)
        ax.axis("off")

        savefig(fig, f"images/rgb_{args.type}_{snap}_{i}")
