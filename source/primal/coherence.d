module primal.coherence;

import std.algorithm : map, sum;
import std.math : sqrt, abs;
import std.range : iota;

/// Computes pairwise correlation magnitude between state vectors.
double coherenceScore(const double[][] states) {
    auto n = states.length;
    if (n == 0) return 0;
    double accum = 0;
    size_t pairs = 0;
    foreach (i; 0 .. n) {
        foreach (j; i + 1 .. n) {
            accum += correlationMagnitude(states[i], states[j]);
            ++pairs;
        }
    }
    return pairs == 0 ? 0 : accum / pairs;
}

/// Computes |corr(a,b)| using dot product normalization.
double correlationMagnitude(const double[] a, const double[] b) {
    assert(a.length == b.length);
    double dot = 0;
    double na = 0;
    double nb = 0;
    foreach (idx; 0 .. a.length) {
        dot += a[idx] * b[idx];
        na += a[idx] * a[idx];
        nb += b[idx] * b[idx];
    }
    if (na == 0 || nb == 0) return 0;
    return abs(dot) / (sqrt(na) * sqrt(nb));
}

unittest {
    double[][] states = [[1, 2, 3], [2, 4, 6], [1, -1, 0]];
    auto c = coherenceScore(states);
    assert(c >= 0 && c <= 1);
}
