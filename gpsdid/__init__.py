"""GPS-DID: doubly robust dose-response estimation for continuous-treatment DID."""
from .gps import NormalGPS, HeteroGBMGPS
from .estimators import DoseResponse, dr_dose_response, hirano_imbens
from .diagnostics import balance_table, balance_ok, grade_overlap, placebo_check, sensitivity
from .gains import kink, gain_matrix

__all__ = ["NormalGPS", "HeteroGBMGPS", "DoseResponse", "dr_dose_response", "hirano_imbens",
           "balance_table", "balance_ok", "grade_overlap", "placebo_check", "sensitivity",
           "kink", "gain_matrix"]
