"""A script to plot the UV and Optical slopes from FLARES."""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

# Define the command line arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--type",
    type=str,
    default="stellar",
    help="The type of data to plot",
)

args = parser.parse_args()

# Define the data file
if args.type == "stellar":
    data_file = "data/pure_stellar_<region>_<snap>.hdf5"
elif args.type == "agn":
    data_file = "data/pure_agn_<region>_<snap>.hdf5"
else:
    data_file = "data/combined_<region>_<snap>.hdf5"

# Get regions and snapshots
regions = [str(reg).zfill(2) for reg in range(40)]
snaps = [
    "005_z010p000",
    "006_z009p000",
    "007_z008p000",
    "008_z007p000",
    "009_z006p000",
    "010_z005p000",
]

# Define the spectra keys we'll read
spectra_keys = ["attenuated", "reprocessed"]

# Get the slopes
uv_slopes = {}
optical_slopes = {}
compacts = {}
# Loop over regions and snapshots getting the data
for reg in regions:
    for snap in snaps:
        with h5py.File(
            data_file.replace("<region>", reg).replace("<snap>", snap), "r"
        ) as hdf:
            for spec in spectra_keys:
                try:
                    uv_slopes.setdefault(spec, {}).setdefault(snap, []).append(
                        hdf["UVSlopes"][spec][...]
                    )
                    optical_slopes.setdefault(spec, {}).setdefault(
                        snap, []
                    ).append(hdf["IRSlopes"][spec][...])
                    compacts.setdefault(spec, {}).setdefault(snap, []).append(
                        hdf["Compactness"][spec]["JWST/NIRCam.F444W"][...]
                    )
                except KeyError as e:
                    print(e)
                except TypeError as e:
                    print(e)

# Plot slopes coloured by compactness
fig, ax = plt.subplots(2, 1, figsize=(6, 8))

# Loop over the spectra
for spec in spectra_keys:
    if spec not in uv_slopes:
        continue
    for snap in snaps:
        ax[0].scatter(
            uv_slopes[spec][snap],
            optical_slopes[spec][snap],
            c=compacts[spec][snap],
            cmap="viridis",
        )
        ax[1].scatter(
            uv_slopes[spec][snap],
            optical_slopes[spec][snap],
            c=compacts[spec][snap],
            cmap="viridis",
        )


# Add colourbars
cbar = fig.colorbar(ax[0].collections[0], ax=ax[0])
cbar.set_label("Compactness")
cbar = fig.colorbar(ax[1].collections[0], ax=ax[1])
cbar.set_label("Compactness")

# Add labels
ax[0].set_xlabel("UV Slope")
ax[0].set_ylabel("Optical Slope")

ax[1].set_xlabel("UV Slope")
ax[1].set_ylabel("Optical Slope")

fig.savefig(f"plots/slopes_{args.type}.png")
