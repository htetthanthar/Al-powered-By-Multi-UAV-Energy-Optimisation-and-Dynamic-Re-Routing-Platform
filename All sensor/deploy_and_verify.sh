#!/usr/bin/env bash
set -e

echo "=========================================================================="
echo "  UAV DYNAMIC ROUTE OPTIMIZATION ENGINE: FULL PRODUCTION RUNBOOK          "
echo "=========================================================================="

# 1. Environment & Pre-Flight Checks
echo "[*] Step 1/6: Checking Python Environment & Hardware Accelerators..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import duckdb; print(f'DuckDB Embedded Data Lake Version: {duckdb.__version__}')"

# 2. Checkpoint & ONNX Engine Generation
echo -e "\n[*] Step 2/6: Compiling Neural Surrogates to FP16 ONNX Runtimes..."
python3 export_onnx_tensorrt.py

# 3. Formal Invariant Verification & PyTest Constraint Suite
echo -e "\n[*] Step 3/6: Executing SMT & TLA+ Formal Verification Suite..."
python3 formal_invariants_verifier.py
python3 verify_tla_spec.py
pytest test_pipeline.py -q

# 4. Spin up Background Telemetry & API Microservices
echo -e "\n[*] Step 4/6: Initializing FastAPI Backend and WebSocket Broadcast Daemon..."
uvicorn server:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
python3 ws_telemetry_server.py &
WS_PID=$!

sleep 2

# 5. Live Simulated Mission Execution & Triage Audit
echo -e "\n[*] Step 5/6: Dispatched Active Multi-UAV Urban Delivery Mission..."
python3 main.py dispatch --wind 3.8

# 6. Post-Flight Blackbox Audit & KPI Verification
echo -e "\n[*] Step 6/6: Running Post-Flight Blackbox Telemetry Lake Diagnostics..."
python3 blackbox_analyzer.py

# Cleanup Background Processes
kill $SERVER_PID $WS_PID 2>/dev/null || true

echo "=========================================================================="
echo "[+] ALL SYSTEM TIERS COMPILED, VERIFIED, AND CERTIFIED FOR BVLOS FLIGHT.  "
echo "=========================================================================="