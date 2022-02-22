#!/usr/bin/env python
# import sys
# sys.path.append('E:\OneDrive\OneDrive - Microsoft\matlab workspace\experiment1\prosail\prosail')
import os, sys

tmp_path = os.path.abspath(__file__)
tmp_path = os.path.dirname(tmp_path)
sys.path.append(tmp_path)
from prosail.spectral_library import get_spectra

spectral_lib = get_spectra()
from prosail.prospect_d import run_prospect
from prosail.sail_model import run_sail, run_prosail, run_thermal_sail
