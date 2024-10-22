"""Script for generating the instrument files for the FLARES LRD analysis."""

import webbpsf
from astropy.cosmology import Planck15 as cosmo
from synthesizer.instruments import FilterCollection
from synthesizer.instruments.instrument import Instrument
from unyt import arcsecond, kpc

snapshots = [
    "005_z010p000",
    "006_z009p000",
    "007_z008p000",
    "008_z007p000",
    "009_z006p000",
    "010_z005p000",
]

# Define the filters
nircam_fs = FilterCollection(
    filter_codes=[
        "JWST/NIRCam.F090W",
        "JWST/NIRCam.F115W",
        "JWST/NIRCam.F140M",
        "JWST/NIRCam.F150W",
        "JWST/NIRCam.F162M",
        "JWST/NIRCam.F182M",
        "JWST/NIRCam.F200W",
        "JWST/NIRCam.F210M",
        "JWST/NIRCam.F250M",
        "JWST/NIRCam.F277W",
        "JWST/NIRCam.F300M",
        "JWST/NIRCam.F335M",
        "JWST/NIRCam.F356W",
        "JWST/NIRCam.F360M",
        "JWST/NIRCam.F410M",
        "JWST/NIRCam.F430M",
        "JWST/NIRCam.F444W",
        "JWST/NIRCam.F460M",
        "JWST/NIRCam.F480M",
    ]
)

miri_fs = FilterCollection(
    filter_codes=[
        "JWST/MIRI.F560W",
        "JWST/MIRI.F770W",
        "JWST/MIRI.F1000W",
        "JWST/MIRI.F1130W",
        "JWST/MIRI.F1280W",
        "JWST/MIRI.F1500W",
        "JWST/MIRI.F1800W",
    ]
)
top_hat = FilterCollection(
    tophat_dict={"UV1500": {"lam_eff": 1500, "lam_fwhm": 300}},
)


if __name__ == "__main__":
    # Set up the PSF dictionaries and webbpsf objects
    nircam_psfs = {}
    miri_psfs = {}
    nc = webbpsf.NIRCam()
    miri = webbpsf.MIRI()

    # Get nircam PSFs
    for nc_filt in nircam_fs.filter_codes:
        nc.filter = nc_filt.split(".")[-1]
        psf = nc.calc_psf(oversample=2)
        nircam_psfs[nc_filt] = psf[0].data

    # Get miri psfs
    for miri_filt in miri_fs.filter_codes:
        miri.filter = miri_filt.split(".")[-1]
        psf = miri.calc_psf(oversample=2)
        miri_psfs[miri_filt] = psf[0].data

    # Define the angular resoltions
    nircam_res = 0.031 * arcsecond
    miri_res = 0.111 * arcsecond

    # Loop over snapshots making the isnturment file for each redshift
    # (If imaging worked in angular coordinates, we could just use the same
    # instrument file for all snapshots)
    for snap in snapshots:
        # Get the redshift
        z = float(snap.split("_")[1].replace("z", "").replace("p", "."))

        # Convert the angular resolutions to physical kpc
        arcsec_to_kpc = (
            cosmo.kpc_proper_per_arcmin(z).to("kpc/arcsec").value * kpc / arcsecond
        )

        # Set up the instruments
        nircam = Instrument(
            "JWST/NIRCam",
            filters=nircam_fs,
            psfs=nircam_psfs,
            resolution=arcsec_to_kpc * nircam_res,
        )
        miri = Instrument(
            "JWST/MIRI",
            filters=miri_fs,
            psfs=miri_psfs,
            resolution=arcsec_to_kpc * miri_res,
        )
        uv = Instrument(
            "UV1500",
            filters=top_hat,
            resolution=2.66 / (1 + z) * kpc,
        )

        # Combine them
        instruments = nircam + miri + uv

        # Save the instruments
        instruments.write_instruments(f"flares_lrd_instruments_{snap}.hdf5")
