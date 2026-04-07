"""
mathfuntion.py  (backward-compatibility shim)
=============================================
All functions have been moved to math_utils.py.
This file re-exports everything so that existing
``from mathfuntion import ...`` statements continue to work.
"""
from math_utils import *   # noqa: F401, F403
