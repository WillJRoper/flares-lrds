"""A script for plotting images."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from unyt import erg, s, cm, Hz


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

# Get the synthesizer data for combined, stellar and agn
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
(
    stellar_fluxes,
    stellar_colors,
    stellar_red1,
    stellar_red2,
    stellar_sizes,
    stellar_masks,
    stellar_indices,
    stellar_images,
) = get_synth_data_with_imgs(data_file, "stellar_" + args.spec_type)
(
    agn_fluxes,
    agn_colors,
    agn_red1,
    agn_red2,
    agn_sizes,
    agn_masks,
    agn_indices,
    agn_images,
) = get_synth_data_with_imgs(data_file, "agn_" + args.spec_type)

# Get the galaxy ids for labelling
gal_ids = get_galaxy_identifiers(args.master, indices)

# How many filters do we have?
nfilt = len(images["010_z005p000"].keys())

# Apply masks to the images
for snap in images:
    for filt in images[snap]:
        images[snap][filt] = images[snap][filt][masks[snap], :, :]
        stellar_images[snap][filt] = stellar_images[snap][filt][
            masks[snap], :, :
        ]
        agn_images[snap][filt] = agn_images[snap][filt][masks[snap], :, :]
    gal_ids[snap] = gal_ids[snap][masks[snap]]

# Convert images to nJy
for snap in images:
    for filt in images[snap]:
        images[snap][filt] = (images[snap][filt] * erg / s / cm**2 / Hz).to(
            "nJy"
        )
        stellar_images[snap][filt] = (
            stellar_images[snap][filt] * erg / s / cm**2 / Hz
        ).to("nJy")
        agn_images[snap][filt] = (
            agn_images[snap][filt] * erg / s / cm**2 / Hz
        ).to("nJy")

# Loop over regions
for snap in images:
    # Are there any LRD images to plot?
    if len(images[snap]["F115W"]) == 0:
        continue

    print(f"Plotting images for {snap} ({len(images[snap]['F115W'])} images)")

    # Loop over images
    for i in range(len(images[snap]["F115W"])):
        # Setup the plot with a row of panels per filter
        fig = plt.figure(figsize=(nfilt * 3.5 + 0.4, 3 * 3.5))
        gs = fig.add_gridspec(
            3,
            nfilt + 1,
            wspace=0.0,
            width_ratios=[10] * nfilt + [1],
            hspace=0.0,
        )
        cax = fig.add_subplot(gs[:, -1])
        axes = [fig.add_subplot(gs[0, i]) for i in range(nfilt)]
        stellar_axes = [fig.add_subplot(gs[1, i]) for i in range(nfilt)]
        agn_axes = [fig.add_subplot(gs[2, i]) for i in range(nfilt)]

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
        for filt, ax, sax, aax in zip(
            images[snap], axes, stellar_axes, agn_axes
        ):
            # Plot the images
            im = ax.imshow(
                images[snap][filt][i, :, :],
                cmap="Greys_r",
                norm=norm,
            )
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(filt)

            # Plot the stellar images
            sax.imshow(
                stellar_images[snap][filt][i, :, :],
                cmap="Greys_r",
                norm=norm,
            )
            sax.axis("off")

            # Plot the agn images
            aax.imshow(
                agn_images[snap][filt][i, :, :],
                cmap="Greys_r",
                norm=norm,
            )
            aax.axis("off")

        # Label the rows
        axes[0].set_xlabel("Combined")
        stellar_axes[0].set_xlabel("Stellar")
        agn_axes[0].set_xlabel("AGN")

        # Add the colorbar
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label("Flux / [nJy]")

        savefig(
            fig,
            f"images/{args.spec_type}/compare_{args.spec_type}_"
            f"{snap}_{'_'.join([str(s) for s in gal_ids[snap][i]])}",
        )
