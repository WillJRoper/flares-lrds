"""A script to compare new and old fluxes."""
import matplotlib.pyplot as plt

from utils import get_master_data, get_synth_data


# Define the master file path
master_path = (
    "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"
)
synth_path = "data/pure_stellar_<region>_<snap>.hdf5"

# Get the synthesizer version
(
    att_fluxes,
    att_colors,
    att_red1,
    att_red2,
    att_sizes,
    att_masks,
    indices,
) = get_synth_data(synth_path, "attenuated")

# Get the master fluxes
master_flux = get_master_data(
    master_path,
    indices,
    key="BPASS_2.2.1/Chabrier300/Flux/DustModelI/JWST/NIRCAM/F444W",
)

# Plot the fluxes
fig, ax = plt.subplots()
sc = ax.scatter(
    master_flux["010_z005p000"]["F444W"],
    att_fluxes["010_z005p000"]["F444W"],
    c=att_colors["010_z005p000"]["F277W_F444W"],
    cmap="viridis",
)
ax.set_xlabel("Master Flux")
ax.set_ylabel("Synthesizer Flux")
cbar = fig.colorbar(sc)
cbar.set_label("F277W - F444W")
fig.savefig("plots/flux_comparison.png", dpi=300, bbox_inches="tight")
