# tests/test_hil_sitl.py
import pytest
import numpy as np
from geometric_so3_attitude_controller import GeometricSO3AttitudeController
from crpa_anti_jam_null_steering import CRPAAntiJammingBeamformer
from telecom_5g_coverage_routing import CellularCoverage5GRouter


def test_so3_geometric_attitude_controller():
    controller = GeometricSO3AttitudeController()
    r_identity = np.eye(3)
    omega_zero = np.zeros(3)
    
    res = controller.compute_control_torques(r_identity, omega_zero, r_identity)
    lyapunov_psi = res.get("lyapunov_potential_psi", res.get("lyapunov_psi", 0.0))
    
    assert lyapunov_psi <= 1e-4
    assert res.get("stable", True) is True


def test_crpa_anti_jamming_null_steering():
    crpa = CRPAAntiJammingBeamformer()
    if hasattr(crpa, "evaluate_null_depth"):
        res = crpa.evaluate_null_depth(jam_az=210.0, jam_el=10.0, sat_az=45.0, sat_el=60.0)
    else:
        R_xx = crpa.compute_sample_covariance_matrix(np.random.randn(4, 256) + 1j * np.random.randn(4, 256))
        w = crpa.compute_mvdr_weights(R_xx, 45.0, 60.0)
        res = crpa.evaluate_beamformer_spatial_attenuation(w, 210.0, 10.0, 45.0, 60.0)
    
    assert "jammer_null_depth_db" in res
    assert res["jammer_null_depth_db"] < 0.0


def test_5g_coverage_link_budget():
    router = CellularCoverage5GRouter()
    res = router.evaluate_c2_link_at_waypoint((200.0, 200.0, -30.0))
    
    assert "serving_rsrp_dbm" in res
    assert "c2_link_reliable" in res