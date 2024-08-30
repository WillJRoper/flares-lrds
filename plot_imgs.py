"""A script for plotting images."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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
        # Setup the plot with a row of panels per filter
        fig = plt.figure(figsize=(nfilt * 3.5 + 0.4, 3.5))
        gs = fig.add_gridspec(
            1, nfilt + 1, wspace=0.0, width_ratios=[10] * nfilt + [1]
        )
        cax = fig.add_subplot(gs[0, -1])
        axes = [fig.add_subplot(gs[0, i]) for i in range(nfilt)]

        # Loop over images collecting the global min and max
        vmin = np.inf
        vmax = -np.inf
        for filt in images[snap]:
            vmin = min(
                vmin, images[snap][filt][i][images[snap][filt][i] > 0].min()
            )
            vmax = max(vmax, images[snap][filt][i].max())

        # Create the norm
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

        # Loop over filters and axes
        for filt, ax in zip(images[snap], axes):
            # Plot the images
            im = ax.imshow(
                images[snap][filt][i, :, :],
                cmap="Greys_r",
                norm=norm,
            )
            ax.axis("off")
            ax.set_title(filt)

        # Add the colorbar
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label("Flux / [erg / s / cm$^2$]")

        savefig(fig, f"images/{args.type}_{snap}_{i}")
