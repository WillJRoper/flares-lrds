"""A script defining the pure stellar emission model used for LRDs in FLARES."""

from unyt import Myr, arcsecond

from synthesizer.emission_models.attenuation.dust import PowerLaw
from synthesizer.emission_models import (
    StellarEmissionModel,
    NebularEmission,
    TransmittedEmission,
    ReprocessedEmission,
    EscapedEmission,
)


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
        # Define the incident models
        young_incident = StellarEmissionModel(
            grid=grid,
            label="young_incident",
            extract="incident",
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_incident = StellarEmissionModel(
            grid=grid,
            label="old_incident",
            extract="incident",
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )
        incident = StellarEmissionModel(
            grid=grid,
            label="incident",
            combine=[young_incident, old_incident],
        )

        # Define the nebular emission models
        young_nebular = NebularEmission(
            grid=grid,
            label="young_nebular",
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_nebular = NebularEmission(
            grid=grid,
            label="old_nebular",
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )
        nebular = StellarEmissionModel(
            grid=grid,
            label="nebular",
            combine=[young_nebular, old_nebular],
        )

        # Define the transmitted models
        young_transmitted = TransmittedEmission(
            grid=grid,
            label="young_transmitted",
            fesc=fesc,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_transmitted = TransmittedEmission(
            grid=grid,
            label="old_transmitted",
            fesc=fesc,
            mask_attr="ages",
            mask_op=">=",
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
            fesc=fesc,
            transmitted=young_transmitted,
            nebular=young_nebular,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_reprocessed = ReprocessedEmission(
            grid=grid,
            label="old_reprocessed",
            fesc=fesc,
            transmitted=old_transmitted,
            nebular=old_nebular,
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )
        reprocessed = StellarEmissionModel(
            grid=grid,
            label="reprocessed",
            combine=[young_reprocessed, old_reprocessed],
        )

        # Define the attenuated models
        young_attenuated_nebular = StellarEmissionModel(
            grid=grid,
            label="young_attenuated_nebular",
            apply_dust_to=young_nebular,
            tau_v="young_tau_v",
            dust_curve=PowerLaw(slope=-1.3),
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        young_attenuated = StellarEmissionModel(
            grid=grid,
            label="young_attenuated",
            apply_dust_to=young_attenuated_nebular,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-0.7),
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_attenuated = StellarEmissionModel(
            grid=grid,
            label="old_attenuated",
            apply_dust_to=old_reprocessed,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-0.7),
            mask_attr="ages",
            mask_op=">=",
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
                    incident,
                    nebular,
                    transmitted,
                    reprocessed,
                    young_attenuated_nebular,
                ],
            )
            return

        # If we have an escape fraction, we need to include the escaped
        # emission
        young_escaped = EscapedEmission(
            grid,
            fesc=fesc,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_escaped = EscapedEmission(
            grid,
            fesc=fesc,
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )
        escaped = StellarEmissionModel(
            grid=grid,
            label="escaped",
            combine=[young_escaped, old_escaped],
        )

        # Define the intrinsc emission (we have this since there is an escape
        # fraction)
        young_intrinsic = StellarEmissionModel(
            grid=grid,
            label="young_intrinsc",
            combine=[young_reprocessed, young_escaped],
        )
        old_intrinsic = StellarEmissionModel(
            grid=grid,
            label="old_intrinsc",
            combine=[old_reprocessed, old_escaped],
        )
        intrinsic = StellarEmissionModel(
            grid=grid,
            label="intrinsic",
            combine=[young_intrinsic, old_intrinsic],
        )

        # Define the attenuated
        attenuated = StellarEmissionModel(
            grid=grid,
            label="attenuated",
            combine=[young_attenuated, old_attenuated],
        )

        # Finaly, combine to get the emergent emission
        StellarEmissionModel.__init__(
            grid=grid,
            label="emergent",
            combine=[escaped, attenuated],
            related_models=[
                incident,
                nebular,
                transmitted,
                reprocessed,
                young_attenuated_nebular,
                intrinsic,
            ],
        )


class FLARESLRDsEmission(StellarEmissionModel):
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
        # Define the incident models
        young_incident = StellarEmissionModel(
            grid=grid,
            label="young_incident",
            extract="incident",
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_incident = StellarEmissionModel(
            grid=grid,
            label="old_incident",
            extract="incident",
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )
        incident = StellarEmissionModel(
            grid=grid,
            label="incident",
            combine=[young_incident, old_incident],
        )

        # Define the nebular emission models
        young_nebular = NebularEmission(
            grid=grid,
            label="young_nebular",
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_nebular = NebularEmission(
            grid=grid,
            label="old_nebular",
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )
        nebular = StellarEmissionModel(
            grid=grid,
            label="nebular",
            combine=[young_nebular, old_nebular],
        )

        # Define the transmitted models
        young_transmitted = TransmittedEmission(
            grid=grid,
            label="young_transmitted",
            fesc=fesc,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_transmitted = TransmittedEmission(
            grid=grid,
            label="old_transmitted",
            fesc=fesc,
            mask_attr="ages",
            mask_op=">=",
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
            fesc=fesc,
            transmitted=young_transmitted,
            nebular=young_nebular,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_reprocessed = ReprocessedEmission(
            grid=grid,
            label="old_reprocessed",
            fesc=fesc,
            transmitted=old_transmitted,
            nebular=old_nebular,
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )
        reprocessed = StellarEmissionModel(
            grid=grid,
            label="reprocessed",
            combine=[young_reprocessed, old_reprocessed],
        )

        # Define the angular aperture reporcessed models
        young_angular_small_reprocessed = ReprocessedEmission(
            grid=grid,
            label="young_0p2_aperture_reprocessed",
            fesc=fesc,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
            nebular=young_nebular,
            transmitted=young_transmitted,
        )
        young_angular_small_reprocessed.add_mask(
            attr="angular_radii",
            op="<=",
            thresh=0.2 * arcsecond,
        )
        old_angular_small_reprocessed = ReprocessedEmission(
            grid=grid,
            label="old_0p2_aperture_reprocessed",
            fesc=fesc,
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
            nebular=old_nebular,
            transmitted=old_transmitted,
        )
        old_angular_small_reprocessed.add_mask(
            attr="angular_radii",
            op="<=",
            thresh=0.2 * arcsecond,
        )
        angular_small_reprocessed = StellarEmissionModel(
            grid=grid,
            label="0p2_aperture_reprocessed",
            combine=[
                young_angular_small_reprocessed,
                old_angular_small_reprocessed,
            ],
        )
        young_angular_large_reprocessed = ReprocessedEmission(
            grid=grid,
            label="young_0p4_aperture_reprocessed",
            fesc=fesc,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
            nebular=young_nebular,
            transmitted=young_transmitted,
        )
        young_angular_large_reprocessed.add_mask(
            attr="angular_radii",
            op="<=",
            thresh=0.4 * arcsecond,
        )
        old_angular_large_reprocessed = ReprocessedEmission(
            grid=grid,
            label="old_0p4_aperture_reprocessed",
            fesc=fesc,
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
            nebular=old_nebular,
            transmitted=old_transmitted,
        )
        old_angular_large_reprocessed.add_mask(
            attr="angular_radii",
            op="<=",
            thresh=0.4 * arcsecond,
        )
        angular_large_reprocessed = StellarEmissionModel(
            grid=grid,
            label="0p4_aperture_reprocessed",
            combine=[
                young_angular_large_reprocessed,
                old_angular_large_reprocessed,
            ],
        )

        # Define the attenuated models
        young_attenuated_nebular = StellarEmissionModel(
            grid=grid,
            label="young_attenuated_nebular",
            apply_dust_to=young_nebular,
            tau_v="young_tau_v",
            dust_curve=PowerLaw(slope=-1.3),
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        young_attenuated = StellarEmissionModel(
            grid=grid,
            label="young_attenuated",
            apply_dust_to=young_attenuated_nebular,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-0.7),
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_attenuated = StellarEmissionModel(
            grid=grid,
            label="old_attenuated",
            apply_dust_to=old_reprocessed,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-0.7),
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )

        # Define the angular aperture attenuated models
        angular_small_attenuated = StellarEmissionModel(
            grid=grid,
            label="0p2_aperture_attenuated",
            apply_dust_to=angular_small_reprocessed,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-0.7),
            mask_attr="angular_radii",
            mask_op="<=",
            mask_thresh=0.2 * arcsecond,
        )
        angular_large_attenuated = StellarEmissionModel(
            grid=grid,
            label="0p4_aperture_attenuated",
            apply_dust_to=angular_large_reprocessed,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-0.7),
            mask_attr="angular_radii",
            mask_op="<=",
            mask_thresh=0.4 * arcsecond,
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
                    incident,
                    nebular,
                    transmitted,
                    reprocessed,
                    young_attenuated_nebular,
                    angular_small_reprocessed,
                    angular_large_reprocessed,
                    angular_small_attenuated,
                    angular_large_attenuated,
                ],
            )
            return

        # If we have an escape fraction, we need to include the escaped
        # emission
        young_escaped = EscapedEmission(
            grid,
            fesc=fesc,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_escaped = EscapedEmission(
            grid,
            fesc=fesc,
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )
        escaped = StellarEmissionModel(
            grid=grid,
            label="escaped",
            combine=[young_escaped, old_escaped],
        )

        # Define the intrinsc emission (we have this since there is an escape
        # fraction)
        young_intrinsic = StellarEmissionModel(
            grid=grid,
            label="young_intrinsic",
            combine=[young_reprocessed, young_escaped],
        )
        old_intrinsic = StellarEmissionModel(
            grid=grid,
            label="old_intrinsic",
            combine=[old_reprocessed, old_escaped],
        )
        intrinsic = StellarEmissionModel(
            grid=grid,
            label="intrinsic",
            combine=[young_intrinsic, old_intrinsic],
        )

        # Define the attenuated
        attenuated = StellarEmissionModel(
            grid=grid,
            label="attenuated",
            combine=[young_attenuated, old_attenuated],
        )

        # Finaly, combine to get the emergent emission
        StellarEmissionModel.__init__(
            grid=grid,
            label="emergent",
            combine=[escaped, attenuated],
            related_models=[
                incident,
                nebular,
                transmitted,
                reprocessed,
                young_attenuated_nebular,
                intrinsic,
                angular_small_reprocessed,
                angular_large_reprocessed,
                angular_small_attenuated,
                angular_large_attenuated,
            ],
        )
