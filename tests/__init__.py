# -*- coding: utf-8 -*-
# System modules
import logging
import os

# External modules

# Internal modules

from . import delag

from . import test_data
from . import test_flow

__all__ = ['delag']

def runtest(module, verbose=False):
    if verbose:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(filename=os.devnull) # discard any logging output
        # logging.basicConfig(level = logging.INFO)
    # run the tests
    module.run()

# run all tests
def runall(verbose=False):
    for module in [delag]:
        runtest(module=module,verbose=verbose)
        print()

