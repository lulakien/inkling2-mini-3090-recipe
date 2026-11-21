# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2: platform-specific implementations live under `nvidia/`; configs,
processing, quantization metadata and the DSpark speculator are shared.

BOOT OVERLAY (sm8x, `~/ai/bench/inkling/work/dspark_decode/boot/`): this copy is
mounted over the image's `vllm/models/inkling2/__init__.py` (a destination the
launcher does not use) and adds, without touching any launcher-mounted file:

* `compute_logits_local` on the target model class(es) - inert unless the
  runner's batch-sharded sampler is enabled with
  `--enable-batch-sharded-sampling`; it returns the TP-local (vocab-sharded)
  logits so the all-gather becomes a half-size all-to-all.
* resolution of `Inkling2DSparkModel` from `vllm.models.inkling2.nvidia.
  dspark_boot`, the int8-draft overlay (`VLLM_INKLING_W8` containing "draft").
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nvidia.dspark import Inkling2DSparkModel
    from .nvidia.loop import Inkling2LoopForConditionalGeneration
    from .nvidia.model import Inkling2ForConditionalGeneration

__all__ = [
    "Inkling2DSparkModel",
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
]


def _compute_logits_local(self, hidden_states):  # noqa: ANN001
    """TP-local logits (no vocab all-gather), as other models implement it."""
    return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)


def _patch_target_logits(cls):
    from vllm.logger import init_logger

    if not hasattr(cls, "compute_logits_local"):
        cls.compute_logits_local = _compute_logits_local
        init_logger(__name__).info_once(
            "sm8x boot overlay: compute_logits_local added to %s", cls.__name__
        )
    return cls


def __getattr__(name: str):
    if name == "Inkling2DSparkModel":
        from vllm.logger import init_logger

        from .nvidia import dspark_boot

        init_logger(__name__).info_once(
            "sm8x boot overlay: draft model from nvidia.dspark_boot"
        )
        return dspark_boot.Inkling2DSparkModel
    if name == "Inkling2LoopForConditionalGeneration":
        from .nvidia import loop, model

        _patch_target_logits(model.Inkling2ForConditionalGeneration)
        return _patch_target_logits(loop.Inkling2LoopForConditionalGeneration)
    if name in __all__:
        from .nvidia import model

        return _patch_target_logits(getattr(model, name))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
