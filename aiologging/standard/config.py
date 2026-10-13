"""Alias for the shared public configuration module."""

import sys
from .. import config

sys.modules[__name__] = config
