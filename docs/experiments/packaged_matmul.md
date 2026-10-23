# Packaged gate/up matmul experiment

This branch adds an explicit, default-off A/B experiment for Qwen3-0.6B,
BF16, TP=1. It selects `VllmPallasMatmulLinearMethod` for prefixes ending in
`.mlp.gate_up_proj`. It does not replace attention or down-projection matmuls.
The shape contract is canonical `[1024, 6144]` weights and BF16 activations.
Other models with the same prefix and dimensions are not identified by model ID;
only Qwen3-0.6B is in the intended validation scope.

## Installation on Colab v5e

Use Python 3.12 and the v0.29.0 dependency set. In that environment, after
cloning this fork and selecting the experiment commit:

```bash
UV_TORCH_BACKEND=cpu uv pip install vllm-torchtpu==0.29.0
python -m pip install /content/pallas_matmul_subset-0.1.0-py3-none-any.whl
# Remove the old external monkey-patch adapter if it was installed.
python -m pip uninstall -y colab-kernel-ab
# Run in this fork's checkout. Dependencies were installed above.
SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0 python -m pip install --no-deps -e .
python -m pip check
python -c 'import vllm_torchtpu; print(vllm_torchtpu.__file__)'
```

The last command must point to this checkout. The SCM override keeps the local
core version aligned with the combined wheel's exact dependency; record the
actual Git commit alongside results. The experimental matmul wheel is a local
artifact, not a published PyPI package. Obtain it from the prior experiment's
`results/matmul_package/dist/`; installing this fork alone does not install it.
The default path does not import this optional package.

## Running baseline and custom

Start separate processes with the same model revision, prompts and settings:

```bash
TPU_QWEN3_CUSTOM_MATMUL=0 python inference.py
TPU_QWEN3_CUSTOM_MATMUL=1 python inference.py
```

`inference.py` denotes your Qwen3 inference script. Keep BF16, TP=1 and no
load-time quantization. Changing the environment variable after constructing a
model does not change its selected methods; restart the process. Custom mode
logs each selected layer during weight loading. Expect 28 gate/up layers for
Qwen3-0.6B. Do not install/enable the previous `colab-kernel-ab` plugin together
with this implementation: that plugin overrides the linear methods at runtime.

## Implementation

- `envs.py`: `TPU_QWEN3_CUSTOM_MATMUL`, default false.
- `quantization/unquantized.py`: selects a subclass at quant-method construction.
  The subclass reuses existing weight transposition, validates the experimental
  envelope, prepares the custom op before compilation, and restores output shape.
- `layers/adapter/pallas_matmul.py`: imports `matmul` from the installed wheel,
  caches `pallas.jax_op`, registers its fake shape implementation, and adapts rows.

The tile/padding policy deliberately matches the earlier experiment: M <= 16
uses block_m=16; larger M uses block_m=128, with padding and output slicing.
This is a reproduction policy, not a tuned performance recommendation. It can
pad M=32/64 to 128. The kernel source is not copied into this repository.

## Validation

In the configured TPU environment, install pytest and run:

```bash
python -m pytest tests/layers/adapter/quantization/test_pallas_matmul.py --use-tpu -q
python -m pytest tests/layers/adapter/quantization/test_unquantized_linear.py --use-tpu -q
```

The new tests cover opt-in scope, conflicting load-time quantization, unsupported
weights, repeated loading, bias/dtype rejection and compiled rank-3 forward
execution for seven row counts. They compare against FP32 matmul of the same
BF16-rounded inputs with atol=rtol=0.02, not bitwise equivalence to all baseline
kernels. The previous independent wheel tests do not establish that this new
fork integration passes or that full model output is equivalent.

Before interpreting performance, rerun the full model A/B on this commit and
check complete generated token sequences as well as TTFT/TPOT/throughput.
Earlier experiments showed differences in generated sequences and slower custom
inference. This change makes no performance or accuracy-equivalence claim.
