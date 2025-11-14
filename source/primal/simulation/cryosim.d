module primal.simulation.cryosim;

import std.algorithm.comparison : clamp, min, max;

import primal.config;

/// Represents organ-specific tolerances for cryogenic revival.
struct OrganProperty {
    string name;            /// organ identifier
    double perfusionRate;   /// normalized perfusion coefficient
    double recoveryRate;    /// normalized recovery gain
    double energyLimit;     /// joules permitted per update
}

/// Maintains the state of the cryogenic simulation.
struct CryoState {
    double temperatureK;    /// average tissue temperature in kelvin
    double vitality;        /// aggregate vitality metric [0,1]
    double quantumEnergy;   /// normalized quantum energy infusion
}

/// Defines canonical organ table derived from the LaTeX appendix.
OrganProperty[] defaultOrgans(const PrimalParameters params) {
    return [
        OrganProperty("brain", 0.002 * params.alpha, 0.007 * 1.47, 3.5 * params.energyBudget),
        OrganProperty("heart", 0.003 * params.alpha, 0.009 * 1.47, 5.0 * params.energyBudget),
        OrganProperty("muscle", 0.004 * params.alpha, 0.011 * 1.47, 7.0 * params.energyBudget)
    ];
}

/// Runs a single update step of the cryogenic revival simulation.
CryoState updateCryoState(const CryoState state,
                          const OrganProperty[] organs,
                          const PrimalParameters params,
                          double dt,
                          double theta,
                          double phi) {
    double totalEnergy = 0;
    foreach (organ; organs) {
        totalEnergy += organ.energyLimit;
    }
    auto newQuantumEnergy = state.quantumEnergy + params.alpha * (theta + phi) * dt;
    auto energyUsed = min(newQuantumEnergy, totalEnergy);
    auto vitalityGain = params.alpha * energyUsed;
    auto vitality = clamp(state.vitality + vitalityGain - params.lambda * state.vitality * dt, 0.0, 1.0);

    auto temperature = max(state.temperatureK + (-0.5 * dt) + 0.01 * energyUsed, 250.0);

    return CryoState(temperature, vitality, newQuantumEnergy - energyUsed * 0.5);
}

unittest {
    auto params = PrimalParameters();
    auto organs = defaultOrgans(params);
    auto state = CryoState(260.0, 0.2, 0.5);
    auto next = updateCryoState(state, organs, params, 0.1, 0.5, 0.2);
    assert(next.vitality >= state.vitality);
}
