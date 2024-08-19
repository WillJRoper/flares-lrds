"""A script defining the pure stellar emission model used for LRDs in FLARES."""
import numpy as np
from unyt import Myr, arcsecond, nm, erg, s, Hz

from synthesizer.emission_models.attenuation.dust import PowerLaw
from synthesizer.emission_models import (
    StellarEmissionModel,
    BlackHoleEmissionModel,
    TemplateEmission,
    NebularEmission,
    TransmittedEmission,
    ReprocessedEmission,
    EscapedEmission,
)
from synthesizer.grid import Template


class AGNTemplateEmission(BlackHoleEmissionModel):
    """
    The stellar emission model used for in FLARES.

    This model is a subclass of the StellarEmissionModel class and is used
    to generate the stellar emission for galaxies in FLARES.
    """

    def __init__(self, agn_template_file):
        """
        Initialize the FLARESLOSEmission model.

        Args:
            grid (Grid): The grid to use for the model.
        """

        # Load the AGN template
        agn_template = np.loadtxt(
            agn_template_file, usecols=(0, 1), skiprows=23
        )

        # Create the Template
        temp = Template(
            lam=agn_template[:, 0] * 0.1 * nm,
            lnu=agn_template[:, 1] * erg / s / Hz,
        )

        # Create the agn template emission model
        agn_intrinsic = TemplateEmission(
            temp,
            emitter="blackhole",
            label="AGN_intrinsic",
        )

        # Define the attenuated AGN model
        BlackHoleEmissionModel.__init__(
            self,
            label="AGN_attenuated",
            apply_dust_to=agn_intrinsic,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-1),
        )
