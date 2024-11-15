"""
A custom tile that implements an asymmetric update rule.
When the weight update in one direction is more steep than in the other (e.g. higher steep in the positive direction),
we can implement a small weight increase as a large increase (due to the high slope)
and small decreases enabled by the lower slope in the other direction.

TODO:
- Create a custom class derived from CustomSimulatorClass (floating point update).
- Check that the update is working correctly.
- Implement mixed-precision.
- Check that implemented mixed-precision gives similar results to the native implementation (DigitalRankUpdate).
"""

from torch import Tensor

from aihwkit.exceptions import TileError
from aihwkit.simulator.tiles.custom import CustomSimulatorTile, CustomRPUConfig, CustomUpdateParameters
from aihwkit.simulator.configs.compounds import MixedPrecisionCompound
from typing import ClassVar, Type
from dataclasses import dataclass, field

class AsymmetricUpdateTile(CustomSimulatorTile):
    """ New class where small updates in a high-slope direction are implemented by overshooting and going down in the other direction. """

    def __init__(self, x_size: int, d_size: int, rpu_config: "AsymmetricUpdateRPUConfig", bias: bool = False):
        super().__init__(x_size, d_size, rpu_config, bias)

    def update(
        self,
        x_input: Tensor,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Update with mixed-precision scheme.

        Note:
            Ignores additional arguments

        Raises:
            TileError: in case transposed input / output or bias is requested
        """
        if bias or in_trans or out_trans or non_blocking:
            raise TileError("transposed inputs or analog bias not supported")

        # ORIGINAL IMPLEMENTATION:
        # delta_w = d_input.view(-1, d_input.size(-1)).T @ x_input.view(-1, x_input.size(-1))

        # if self._update.gradient_noise:
        #     delta_w += self._update.gradient_noise * randn_like(delta_w)

        # self._analog_weight = self._analog_weight - self.learning_rate * delta_w  # type: ignore
        # END ORIGINAL IMPLEMENTATION

@dataclass
class AsymmetricUpdateRPUConfig(CustomRPUConfig):
    """ Configuration for the AsymmetricUpdateTile. """

    simulator_tile_class: ClassVar[Type] = AsymmetricUpdateTile
    """Simulator tile class implementing the analog forward / backward / update."""

    device: MixedPrecisionCompound = field(default_factory=MixedPrecisionCompound)
    """Parameter that modify the behavior of the pulsed device."""