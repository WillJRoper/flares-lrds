"""A script defining the pure stellar emission model used for LRDs in FLARES."""

from synthesizer.emission_models import (
    NebularEmission,
    ReprocessedEmission,
    StellarEmissionModel,
    TransmittedEmission,
)
from synthesizer.emission_models.attenuation.dust import PowerLaw
from unyt import Myr


class FLARESLOSEmission(StellarEmissionModel):
    """
    The stellar emission model used for in FLARES.

    This model is a subclass of the StellarEmissionModel class and is used
    to generate the stellar emission for galaxies in FLARES.
    """

    def __init__(self, grid, fesc=0.0, fesc_ly_alpha=1.0):
        """
        Initialize the FLARESLOSEmission model.

        Args:
            grid (Grid): The grid to use for the model.
        """
        # Define the nebular emission models
        nebular = NebularEmission(
            grid=grid,
            label="nebular",
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )

        # Define the transmitted models
        young_transmitted = TransmittedEmission(
            grid=grid,
            label="young_transmitted",
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )
        old_transmitted = TransmittedEmission(
            grid=grid,
            label="old_transmitted",
            mask_attr="ages",
            mask_op=">",
            mask_thresh=10 * Myr,
        )
        transmitted = StellarEmissionModel(
            grid=grid,
            label="transmitted",
            combine=[young_transmitted, old_transmitted],
        )

        # Define the reprocessed models
        young_reprocessed = ReprocessedEmission(
            grid=grid,
            label="young_reprocessed",
            transmitted=young_transmitted,
            nebular=nebular,
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )
        reprocessed = StellarEmissionModel(
            grid=grid,
            label="reprocessed",
            combine=[young_reprocessed, old_transmitted],
        )

        # Define the attenuated models
        young_attenuated_nebular = StellarEmissionModel(
            grid=grid,
            label="young_attenuated_nebular",
            apply_dust_to=young_reprocessed,
            tau_v="young_tau_v",
            dust_curve=PowerLaw(slope=-1),
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )
        young_attenuated = StellarEmissionModel(
            grid=grid,
            label="young_attenuated",
            apply_dust_to=young_attenuated_nebular,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-1),
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )
        old_attenuated = StellarEmissionModel(
            grid=grid,
            label="old_attenuated",
            apply_dust_to=old_transmitted,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-1),
            mask_attr="ages",
            mask_op=">",
            mask_thresh=10 * Myr,
        )

        # If we have no escape fraction, we can just combine the models and
        # get our final attenuated emission
        if fesc == 0.0:
            StellarEmissionModel.__init__(
                self,
                grid=grid,
                label="attenuated",
                combine=[young_attenuated, old_attenuated],
                related_models=[
                    nebular,
                    transmitted,
                    reprocessed,
                    young_attenuated_nebular,
                ],
            )
            self.set_per_particle(True)
            return

        raise NotImplementedError(
            "The FLARESLOSEmission model is not yet implemented with fesc > 0."
        )
