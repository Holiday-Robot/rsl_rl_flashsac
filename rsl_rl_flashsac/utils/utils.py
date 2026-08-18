# Copyright (c) 2021-2026, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, Holiday Robotics
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from RSL-RL Project (BSD-3-Clause license),
# with modifications by Holiday Robotics (BSD-3-Clause license).

"""Helpers bridging rsl_rl_flashsac class resolution with upstream rsl_rl."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import torch
from rsl_rl.utils import resolve_callable as _rsl_resolve_callable

_LOCAL_MODULES = (
    "rsl_rl_flashsac.algorithms",
    "rsl_rl_flashsac.models",
    "rsl_rl_flashsac.modules",
    "rsl_rl_flashsac.runners",
    "rsl_rl_flashsac.storage",
)


def resolve_callable(callable_or_name: type | Callable | str) -> Callable:
    """Resolve a callable from a string, type, or return the callable directly.

    Simple names (e.g. ``"FlashSAC"``, ``"FlashSACActor"``) are first looked up in the
    ``rsl_rl_flashsac`` subpackages; anything else (qualified names, upstream rsl_rl simple
    names, or callables) is delegated to :func:`rsl_rl.utils.resolve_callable`.
    """
    if isinstance(callable_or_name, str) and "." not in callable_or_name and ":" not in callable_or_name:
        for module_name in _LOCAL_MODULES:
            module = importlib.import_module(module_name)
            if hasattr(module, callable_or_name):
                return getattr(module, callable_or_name)  # type: ignore[no-any-return]
    return _rsl_resolve_callable(callable_or_name)


def resolve_compile_mode(mode: str) -> str:
    """Resolve 'auto' compile mode based on the installed torch version.

    'auto' picks autotuned kernels but disables CUDA graphs: inside Isaac Sim processes,
    CUDA-graph-backed compile modes crash with cudagraph-trees pool bookkeeping errors
    (not reproducible standalone). Set compile_mode="max-autotune" explicitly to opt into
    CUDA graphs outside Isaac Sim.
    """
    if mode != "auto":
        return mode
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        return "max-autotune-no-cudagraphs"
    return "default"
