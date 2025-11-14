module primal.config;

import std.datetime : Duration, msecs;

/// Holds the primary PRIMAL LOGIC parameters.
struct PrimalParameters {
    double alpha = 0.54;          /// dimensionless control gain (from training window [0.52, 0.56])
    double lambda = 0.115;        /// decay rate in 1/ms controlling dissipation
    double epsilon = 0.1;         /// imaginary component weighting
    double gamma = 0.2;           /// plasma coupling coefficient
    double couplingStrength = 0.5;/// spatial coupling constant K in 1/mm^2
    double sigma = 0.15;          /// coherence scaling coefficient
    double phaseCoupling = 0.25;  /// phase synchronization gain
    double collapseThreshold = 2.0;/// threshold for quantum collapse probability (dimensionless)
    double superpositionStrength = 0.3; /// amplitude mixing coefficient
    double energyBudget = 1.0;    /// joules per update (normalized)
    double consensusThreshold = 0.8; /// multi-agent consensus target
}

/// Sampling windows for temporal dynamics in milliseconds.
immutable double[] temporalScalesMs = [0.1, 1.0, 10.0, 100.0];

/// Spatial scales in millimetres used for multi-scale coupling.
immutable double[] spatialScalesMm = [0.1, 1.0, 5.0, 20.0];

/// Convenience accessor returning the canonical control update period.
Duration controlPeriod() {
    // The LaTeX specification used 50 microsecond loops; we approximate with milliseconds for practicality.
    return msecs(1);
}
