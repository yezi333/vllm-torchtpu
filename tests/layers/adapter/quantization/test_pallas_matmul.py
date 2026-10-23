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
"""Opt-in routing and real TPU forward checks for packaged matmul."""

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmPallasMatmulLinearMethod,
    VllmUnquantizedConfig,
    VllmUnquantizedLinearMethod,
)


@pytest.mark.parametrize(
    "enabled,prefix,expected",
    [
        ("0", "model.layers.0.mlp.gate_up_proj", VllmUnquantizedLinearMethod),
        ("1", "model.layers.0.mlp.down_proj", VllmUnquantizedLinearMethod),
        ("1", "model.layers.0.self_attn.qkv_proj", VllmUnquantizedLinearMethod),
        ("1", "model.layers.0.mlp.gate_up_proj", VllmPallasMatmulLinearMethod),
    ],
)
def test_opt_in_dispatch(monkeypatch, enabled, prefix, expected):
    monkeypatch.setenv("TPU_QWEN3_CUSTOM_MATMUL", enabled)
    monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "")
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    method = VllmUnquantizedConfig().get_quant_method(layer, prefix)
    assert type(method) is expected


def test_rejects_load_time_quantization(monkeypatch):
    monkeypatch.setenv("TPU_QWEN3_CUSTOM_MATMUL", "1")
    monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "gate_up_proj")
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    with pytest.raises(ValueError, match="load-time quantization"):
        VllmUnquantizedConfig().get_quant_method(
            layer, "model.layers.0.mlp.gate_up_proj"
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("rows", [1, 16, 32, 64, 127, 128, 256])
def test_compiled_forward_and_reload(device, rows):
    if device.type != "tpu":
        pytest.skip("Run with --use-tpu and install pallas-matmul-subset wheel")
    torch.manual_seed(rows)
    layer = torch.nn.Module()
    layer.tp_size = 1
    layer.prefix = "model.layers.0.mlp.gate_up_proj"
    original = torch.randn(6144, 1024, dtype=torch.bfloat16, device=device) * 0.02
    layer.weight = torch.nn.Parameter(original, requires_grad=False)
    method = VllmPallasMatmulLinearMethod()
    method.process_weights_after_loading(layer)
    first_op = method._matmul_op
    method.process_weights_after_loading(layer)
    assert method._matmul_op is first_op
    assert layer.weight.shape == (1024, 6144)
    x = torch.randn(1, rows, 1024, dtype=torch.bfloat16, device=device)

    def forward(x):
        return method.apply(layer, x)

    compiled = torch.compile(forward, backend="tpu", fullgraph=True)
    actual = compiled(x).float().cpu()
    reference = (x.float() @ layer.weight.float()).cpu()
    assert actual.shape == (1, rows, 6144)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)
    torch.testing.assert_close(compiled(x).float().cpu(), actual, atol=0, rtol=0)
    with pytest.raises(ValueError, match="bias"):
        method.apply(layer, x, torch.zeros(6144, device=device))
    with pytest.raises(ValueError, match="BF16 input"):
        method.apply(layer, x.float())


@pytest.mark.parametrize(
    "shape,dtype,tp",
    [
        ((32, 64), torch.bfloat16, 1),
        ((6144, 1024), torch.float32, 1),
        ((6144, 1024), torch.bfloat16, 2),
    ],
)
def test_rejects_unsupported_weights(shape, dtype, tp):
    layer = torch.nn.Module()
    layer.tp_size = tp
    layer.weight = torch.nn.Parameter(torch.empty(shape, dtype=dtype))
    with pytest.raises(ValueError, match="requires TP=1"):
        VllmPallasMatmulLinearMethod().process_weights_after_loading(layer)
