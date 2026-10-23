# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bridge to the optional, separately installed experimental matmul wheel."""

from collections.abc import Callable
from threading import Lock

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

_op: Callable | None = None
_lock = Lock()


def get_pallas_matmul_op() -> Callable:
    """Register once during weight loading, before torch compilation.

    The optional pallas-matmul-subset wheel must be installed when enabled.
    """
    global _op
    with _lock:
        if _op is not None:
            return _op
        from pallas_matmul_subset import matmul

        def matmul_jax(x: jax.Array, weight: jax.Array) -> jax.Array:
            rows = x.shape[0]
            # Preserve the earlier A/B experiment's tile and padding policy.
            block_m = 16 if rows <= 16 else 128
            padded_rows = ((rows + block_m - 1) // block_m) * block_m
            padded = jnp.pad(x, ((0, padded_rows - rows), (0, 0)))
            return matmul(padded, weight, block_shape=(block_m, 128), block_k=128)[
                :rows
            ]

        op = pallas.jax_op("pallas::experimental_gate_up_matmul", matmul_jax)

        def fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.empty(
                (x.shape[0], weight.shape[1]), dtype=x.dtype, device=x.device
            )

        op.register_fake(fake)
        _op = op
        return op
