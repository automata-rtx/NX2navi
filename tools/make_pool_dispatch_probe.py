"""Emit the compute stage used by the ordered pooled DispatchIndirect probe."""
from make_pool_restore_probe import build_module, header


HEADER = header(build_module(compute=True), 'pool_dispatch_probe', 'make_pool_dispatch_probe.py')
