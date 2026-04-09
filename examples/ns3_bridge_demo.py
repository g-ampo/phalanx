"""Phalanx ns-3 bridge demo (stub).

Shows how ns-3 integration works through the NS3Bridge API,
including prediction-augmented scheduling and metric collection.
This runs in stub mode -- ns-3 does not need to be installed.
When ns-3 and ns3-ai are available, the bridge connects a Phalanx
scheduler to a live ns-3 simulation.
"""
from phalanx.ns3 import NS3Bridge
from phalanx.schedulers import LyapunovDPP
from phalanx.predictors import PersistencePredictor
from phalanx.metrics import CostMetric, PredictionErrorMetric

M, D_OBS, F = 8, 4, 4

# ====================================================================
# Step 1: Create components
# ====================================================================
scheduler = LyapunovDPP(V=5.0, F=F, J_total=0.5)
predictor = PersistencePredictor(d_obs_total=M * D_OBS)
metric = CostMetric()
print(f"Scheduler: {scheduler.name}")

# ====================================================================
# Step 2: Create and configure the bridge
# ====================================================================
bridge = NS3Bridge(env_id="phalanx-v0", M=M, d_obs=D_OBS, F=F)
bridge.register_scheduler(scheduler)
bridge.register_predictor(predictor, obs_window_size=20)
bridge.register_metric(metric)

# ====================================================================
# Step 3: Run (stub mode without ns-3, full mode with ns3-ai)
# ====================================================================
result = bridge.run(T=1000, seed=42)
print(f"Mode: {result['mode']}")
print(f"Predictor: {result['predictor']}")
if "summary" in result:
    print(f"Summary: {result['summary']}")

# ====================================================================
# Step 4: Clean up
# ====================================================================
bridge.close()
print("Bridge closed.")

# ====================================================================
# Prediction error tracking example
# ====================================================================
print("\n--- With prediction error metric ---")
bridge2 = NS3Bridge(env_id="phalanx-v0", M=M, d_obs=D_OBS, F=F)
bridge2.register_scheduler(LyapunovDPP(V=5.0, F=F, J_total=0.5))
bridge2.register_predictor(PersistencePredictor(d_obs_total=M * D_OBS))
bridge2.register_metric(PredictionErrorMetric())
result2 = bridge2.run(T=1000, seed=42)
print(f"Prediction NMSE: {result2['summary']['nmse']:.4f}")
bridge2.close()
