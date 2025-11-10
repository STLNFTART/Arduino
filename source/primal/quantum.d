module primal.quantum;

import std.algorithm : clamp;
import std.exception : enforce;
import std.math : cos, sin, PI, sqrt, abs;
import std.random : uniform01;
import std.typecons : Tuple;

import primal.config;

/// Represents the complex-valued quantum-inspired state field ψ(x,y,t).
struct QuantumState {
    double[][] realPart; /// ψ_r grid sampled in SI units (normalized amplitude)
    double[][] imagPart; /// ψ_i grid sampled in SI units (normalized amplitude)
}

/// Initializes a rectangular grid filled with zeros.
QuantumState makeQuantumState(size_t nx, size_t ny) {
    enforce(nx > 1 && ny > 1, "Grid must be at least 2x2 for Laplacian.");
    double[][] realPart;
    double[][] imagPart;
    realPart.length = nx;
    imagPart.length = nx;
    foreach (i; 0 .. nx) {
        realPart[i].length = ny;
        imagPart[i].length = ny;
    }
    return QuantumState(realPart, imagPart);
}

/// Applies a discrete 5-point Laplacian (Δψ) with Neumann boundaries.
private double laplacianAt(const double[][] field, size_t ix, size_t iy, double dx) {
    auto nx = field.length;
    auto ny = field[0].length;
    immutable left = (ix == 0) ? field[ix][iy] : field[ix - 1][iy];
    immutable right = (ix + 1 == nx) ? field[ix][iy] : field[ix + 1][iy];
    immutable down = (iy == 0) ? field[ix][iy] : field[ix][iy - 1];
    immutable up = (iy + 1 == ny) ? field[ix][iy] : field[ix][iy + 1];
    immutable center = field[ix][iy];
    return (left + right + up + down - 4.0 * center) / (dx * dx);
}

/// Time-steps the quantum state according to the LaTeX specification.
QuantumState evolveQuantumState(const QuantumState state,
                                const PrimalParameters params,
                                double gammaField,
                                double thetaSignal,
                                double phiSignal,
                                double dt,
                                double dx,
                                double plasmaGamma,
                                ref double collapseProbability) {
    auto nx = state.realPart.length;
    auto ny = state.realPart[0].length;
    auto next = makeQuantumState(nx, ny);

    double interferenceAccumulator = 0.0;

    foreach (ix; 0 .. nx) {
        foreach (iy; 0 .. ny) {
            auto lapR = laplacianAt(state.realPart, ix, iy, dx);
            auto lapI = laplacianAt(state.imagPart, ix, iy, dx);

            auto dpsiR = -params.lambda * state.realPart[ix][iy]
                        + params.couplingStrength * lapR
                        + params.gamma * gammaField
                        + params.alpha * thetaSignal;

            auto dpsiI = -params.lambda * state.imagPart[ix][iy]
                        + params.couplingStrength * lapI
                        + params.gamma * plasmaGamma
                        + params.alpha * phiSignal
                        + params.epsilon * state.realPart[ix][iy];

            next.realPart[ix][iy] = state.realPart[ix][iy] + dt * dpsiR;
            next.imagPart[ix][iy] = state.imagPart[ix][iy] + dt * dpsiI;

            interferenceAccumulator += next.realPart[ix][iy] * next.imagPart[ix][iy];
        }
    }

    auto amplitude = sqrt(abs(interferenceAccumulator));
    collapseProbability = clamp(amplitude / params.collapseThreshold, 0.0, 1.0);
    return next;
}

/// Creates a time-varying α mixing pair following the quantum superposition algorithm.
Tuple!(double, double) computeSuperposition(double alpha, double timeMs) {
    immutable omega = 2.0 * PI / 10.0; // 10 ms oscillation per specification
    auto aReal = alpha * cos(omega * timeMs);
    auto aImag = alpha * sin(omega * timeMs);
    return typeof(return)(aReal, aImag);
}

/// Computes a randomized collapse event.
bool shouldCollapse(ref double probability) {
    auto r = uniform01!double();
    if (r < probability) {
        probability = 0.0;
        return true;
    }
    probability *= 0.95; // decay probability slightly when collapse does not occur
    return false;
}

unittest {
    auto params = PrimalParameters();
    auto state = makeQuantumState(4, 4);
    double collapseProb = 0.0;
    auto evolved = evolveQuantumState(state, params, 0.1, 1.0, 0.5, 0.01, 1.0, 0.2, collapseProb);
    assert(evolved.realPart[0][0] != state.realPart[0][0], "State should evolve under forcing");
    assert(collapseProb >= 0.0 && collapseProb <= 1.0);

    auto mix = computeSuperposition(0.54, 1.0);
    assert(mix[0] != mix[1]);
}
