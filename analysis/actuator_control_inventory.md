# Actuator, Motor, and Control Loop Inventory

| File path | Class / Function | Dynamics step method | Arguments (signature) | dt handling |
|-----------|------------------|----------------------|-----------------------|-------------|
| `primal_logic/hand.py` | `RoboticHand` | `step(desired_angles, theta, coherence, step)` | `desired_angles` (list of target angles per joint), `theta` (command envelope), `coherence` (field coherence), `step` (discrete time index) | Uses `self.dt` for Euler integration of joint angles/velocities. |
| `primal_logic/heart_model.py` | `MultiHeartModel` | `step(cardiac_input, brain_setpoint, theta=1.0)` | `cardiac_input` (external drive), `brain_setpoint` (setpoint), `theta` (RPO command envelope) | Uses `self.dt` to Euler-integrate coupled brain/heart potentials. |
| `primal_logic/field.py` | `PrimalLogicField` | `step(theta: float)` | `theta` (command envelope) | Integrates field with fixed `DT` constant per tick. |
| `primal_logic/rpo.py` | `RecursivePlanckOperator` | `step(theta, input_value, step_index)` | `theta` (command envelope), `input_value` (error/control input), `step_index` (discrete index for resonance) | Uses `self.dt` internally for recursive update scaling. |
| `primal_logic/motorhand_integration.py` | `UnifiedPrimalLogicController` | `step(target_angles: Optional[np.ndarray] = None)` | `target_angles` (optional joint targets) | Uses controller `dt` attribute (0.01 s) to drive hand hardware loop. |
