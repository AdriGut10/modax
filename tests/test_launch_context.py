"""Kernels compiled under one CUDA context launch on XLA's stream in another.

A process driving several GPUs compiles a kernel once and launches it on every
GPU. With a ``CUfunction`` bound to the compile-time context, every GPU but one
failed with ``CUDA_ERROR_INVALID_HANDLE`` (400). One GPU reproduces this: the
kernel is compiled while a second, non-primary context is current, and JAX then
launches it in the primary context.
"""

import ctypes

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba_cuda_mlir import cuda

from modax import _jax_numba_custom_call as bridge

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture
def foreign_context():
    """Wrap compile_kernel so every compile runs under a second context on device 0."""

    libcuda = ctypes.CDLL("libcuda.so.1")
    assert libcuda.cuInit(0) == 0
    device = ctypes.c_int()
    assert libcuda.cuDeviceGet(ctypes.byref(device), 0) == 0
    context = ctypes.c_void_p()
    if hasattr(libcuda, "cuCtxCreate_v4"):  # CUDA 13 drivers
        err = libcuda.cuCtxCreate_v4(ctypes.byref(context), None, 0, device)
    else:
        err = libcuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
    assert err == 0
    assert libcuda.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p())) == 0
    yield libcuda, context
    libcuda.cuCtxDestroy_v2(context)


def test_a_kernel_compiled_in_another_context_launches(foreign_context, monkeypatch):
    from modax.rodas5P import _make_jax_launch
    from modax.rodas5P import solve as rodas5p_solve

    libcuda, context = foreign_context
    compiled = []
    original = bridge.compile_kernel

    def compile_kernel(kernel, argtypes):
        assert libcuda.cuCtxPushCurrent_v2(context) == 0
        try:
            compiled.append(original(kernel, argtypes))
        finally:
            assert libcuda.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p())) == 0
        return compiled[-1]

    monkeypatch.setattr(bridge, "compile_kernel", compile_kernel)
    _make_jax_launch.cache_clear()  # compile here, not reuse an earlier test's kernel

    def decay(y, t, p):
        return (-p[0] * y[0],)

    t_span = jnp.linspace(0.0, 2.0, 5)
    y0 = jnp.asarray([[1.5], [0.5]])
    rates = jnp.asarray([[0.7], [2.0]])
    final = rodas5p_solve(decay, y0, t_span, rates, rtol=1e-10, atol=1e-12)[:, -1, 0]
    _make_jax_launch.cache_clear()  # later tests must not launch through this context's compile

    assert compiled, "the solve did not compile a kernel"
    exact = np.asarray(y0[:, 0] * jnp.exp(-rates[:, 0] * t_span[-1]))
    np.testing.assert_allclose(np.asarray(final), exact, rtol=1e-7)
