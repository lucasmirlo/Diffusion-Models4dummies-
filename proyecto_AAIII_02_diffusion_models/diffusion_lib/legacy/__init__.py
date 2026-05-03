# -*- coding: utf-8 -*-
"""Legacy simulation API (pre-diffusion_lib).

These classes and functions provide the original trajectory-simulation interface
used in the early demo notebooks.  They have a fundamentally different contract
from the generation-focused ``diffusion_lib`` core: ``euler_maruyama_integrator``
returns a full ``(times, x_t)`` trajectory array rather than a single denoised
sample, and ``GaussianDiffussionProcess`` parameterises the process via callable
coefficients rather than schedule objects.

Import pattern (replacing the old root-level ``diffusion_process.py``)::

    from diffusion_lib.legacy import diffusion_process as dfp
"""

from . import diffusion_process

__all__ = ["diffusion_process"]
