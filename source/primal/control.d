module primal.control;

import std.algorithm : clamp;
import std.math : cos, exp;

import primal.config;
import primal.coherence;

/// Maintains adaptive α dynamics across multi-scale temporal windows.
double adaptiveAlpha(const PrimalParameters params,
                     double baseAlpha,
                     double averageEnergy,
                     double coherence,
                     double timeStep,
                     const double[] temporalScales = temporalScalesMs,
                     const double[] spatialScales = spatialScalesMm) {
    double baseOscillation = baseAlpha * (1.0 + params.sigma * cos(timeStep * 0.001));
    double energyScaling = baseAlpha * (averageEnergy / (1000.0 * params.energyBudget));
    double coherenceFactor = baseAlpha * params.phaseCoupling * coherence;

    double temporalInfluence = 0;
    foreach (i, scale; temporalScales) {
        auto weight = spatialScales[min(i, spatialScales.length - 1)] / 20.0;
        temporalInfluence += weight * cos(timeStep / scale);
    }

    auto alpha = baseOscillation + energyScaling + coherenceFactor + (temporalInfluence * 0.1);
    return clamp(alpha, 0.52, 0.56);
}

/// Exponential memory integral for control loop (autonomous vehicle equation).
double exponentialMemoryControl(double previousIntegral,
                                 double error,
                                 double lambda,
                                 double theta,
                                 double dt) {
    auto decay = exp(-lambda * dt);
    return previousIntegral * decay + theta * error * dt;
}

unittest {
    auto params = PrimalParameters();
    auto alpha = adaptiveAlpha(params, params.alpha, 5.0, 0.8, 10.0);
    assert(alpha >= 0.52 && alpha <= 0.56);
    auto integral = exponentialMemoryControl(0.0, 0.1, params.lambda, params.alpha, 0.5);
    assert(integral != 0);
}
