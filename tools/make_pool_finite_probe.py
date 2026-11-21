"""Build the fragment shader for the native finite bindless descriptor probe."""
from make_pool_restore_probe import build_module, header

HEADER = header(build_module(finite=True), 'pool_finite_probe', 'make_pool_finite_probe.py')
