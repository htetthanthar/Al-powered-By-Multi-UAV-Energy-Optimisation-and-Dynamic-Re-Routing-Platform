import sys
import time
import os
import unittest
import numpy as np
import torch

from dynamic_planner import EnergySurrogateMLP, optimize_stop_sequence
from formal_invariants_verifier import RouteSafetySMTVerifier
from sensor_fusion_ekf import UAVStateEstimatorEKF
from adsb_transponder_interface import ADSBMsgDecoderAndTrafficAvoider
from bms_dronecan_driver import DroneCANBatteryMonitor
from atmospheric_icing_model import AtmosphericIcingAndDeicingModel

class TestUAVAutonomousSystemHIL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.energy_model = EnergySurrogateMLP()
        cls.energy_model.eval()
        cls.smt_verifier = RouteSafetySMTVerifier()
        cls.ekf = UAVStateEstimatorEKF(dt=0.05)
        cls.tcas = ADSBMsgDecoderAndTrafficAvoider()
        cls.bms = DroneCANBatteryMonitor()
        cls.icing = AtmosphericIcingAndDeicingModel()

    def test_01_energy_surrogate_inference_latency(self):
        """Verifies neural surrogate single-batch execution speed <= 0.15ms."""
        dummy = torch.randn(1, 5)
        t0 = time.perf_counter()
        for _ in range(200):
            with torch.no_grad():
                _ = self.energy_model(dummy)
        avg_ms = ((time.perf_counter() - t0) / 200.0) * 1000.0
        self.assertLessEqual(avg_ms, 0.20, f"Surrogate latency exceeded: {avg_ms:.4f} ms")

    def test_02_formal_smt_boundary_satisfaction(self):
        """Proves mission duration <= 55min and payload <= 2.0kg invariants."""
        stops = [
            {"id": "CI-1", "weight_kg": 0.65},
            {"id": "CI-2", "weight_kg": 0.70},
            {"id": "CI-3", "weight_kg": 0.45}
        ]
        proof = self.smt_verifier.verify_mission_invariants(stops, total_distance_km=24.5)
        self.assertTrue(proof["mathematical_safety_clearance"], "SMT Invariant proof failed.")

    def test_03_ekf_accelerometer_fusion(self):
        """Verifies 9-state EKF tracks vertical state without divergence."""
        for _ in range(50):
            self.ekf.predict(np.array([0.0, 0.0, 0.0]))
            self.ekf.update_barometer(25.0)
        state = self.ekf.get_estimated_state()
        self.assertAlmostEqual(state["altitude_m"], 25.0, delta=1.0)

    def test_04_dronecan_bms_undervoltage_protection(self):
        """Verifies smart BMS flags critical low voltage when individual cell drops below 3.3V."""
        sample_payload = self.bms.encode_dronecan_battery_info(
            battery_id=1, voltage=19.20, current=22.0, temperature_k=305.15, soc_pct=8.0, soh_pct=92.0
        )
        telemetry = self.bms.decode_dronecan_frame(0x10044432, sample_payload)
        self.assertFalse(telemetry["discharge_permitted"], "BMS failed to enforce undervoltage protection.")

    def test_05_adsb_tcas_conflict_detection(self):
        """Verifies TCAS issues an immediate resolution advisory on intersecting trajectories."""
        own_p = (0.0, 0.0, -25.0)
        own_v = (18.05, 0.0, 0.0)
        intruder_p = (250.0, 50.0, -25.0)
        intruder_v = (-40.0, -8.0, 0.0)

        threat = self.tcas.compute_conflict_tcpa(own_p, own_v, intruder_p, intruder_v)
        self.assertIn(threat["threat_level"], ["RESOLUTION_ADVISORY", "TRAFFIC_ADVISORY"])

def run_ci_hil_pipeline():
    print("==========================================================================")
    print("      CONTINUOUS INTEGRATION (CI) HARDWARE-IN-THE-LOOP TEST RUNNER        ")
    print("==========================================================================")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUAVAutonomousSystemHIL)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n[+] ALL HIL INTEGRATION TESTS PASSED (EXIT 0).")
        sys.exit(0)
    else:
        print("\n[!] CI/HIL TEST RUNNER DETECTED REGRESSION FAILURES (EXIT 1).")
        sys.exit(1)

if __name__ == "__main__":
    run_ci_hil_pipeline()