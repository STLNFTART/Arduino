module primal.plasma;

import std.algorithm : map, sum;
import std.math : exp, sqrt, abs;
import std.range : iota;

import primal.config;

/// Represents a scalar plasma field Γ with spatial coupling.
struct PlasmaField {
    double[] values;  /// field samples (normalized energy density)
    double spacingMm; /// physical spacing between samples
}

/// Evolves the plasma field using a diffusion-like process with collective coupling.
void updatePlasmaField(ref PlasmaField field, const PrimalParameters params, double dt) {
    auto n = field.values.length;
    if (n == 0) return;

    double[] next;
    next.length = n;

    foreach (i; 0 .. n) {
        double lap = 0;
        if (i > 0) lap += field.values[i - 1]; else lap += field.values[i];
        if (i + 1 < n) lap += field.values[i + 1]; else lap += field.values[i];
        lap -= 2.0 * field.values[i];
        lap /= field.spacingMm * field.spacingMm;

        double collective = 0;
        foreach (j; 0 .. n) {
            auto dist = abs((cast(double)i - j) * field.spacingMm);
            auto coupling = params.gamma * exp(-dist / 5.0); // r_coupling = 5 mm
            collective += coupling * field.values[j];
        }

        next[i] = field.values[i]
                 + dt * (params.alpha * lap + collective);
    }

    field.values = next;
}

/// Computes the spatial energy stored in the plasma field.
double plasmaEnergy(const PlasmaField field) {
    if (field.values.length == 0) return 0;
    auto energy = field.values.map!(a => a * a).sum;
    return energy * field.spacingMm; // approximate integral via rectangle rule
}

unittest {
    auto params = PrimalParameters();
    PlasmaField field;
    field.values = [0.5, 0.6, 0.4, 0.1];
    field.spacingMm = 1.0;
    updatePlasmaField(field, params, 0.01);
    assert(plasmaEnergy(field) > 0);
}
