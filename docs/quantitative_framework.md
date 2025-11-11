# Quantitative Framework: Lightfoot and Donte Constants with Recursive Planck Extension

The Quantro–Primal formalism uses physical–mathematical constants that normalize kernel energy,
time symmetry, and causal damping.

## Lightfoot and Donte Constants

Two empirical constants anchor normalization:

- **Donte's Constant** (𝓓 ≈ 149.9992314): A scale factor analogous to Planck’s constant *h* but
  renormalized for causal discrete systems. It defines a transformation between information and
  energetic domains using `E_quantro = (𝓓 / 2π) · ω_eff`, where ω_eff is the effective angular
  frequency of heart–brain resonance. This couples energetic stability to informational bandwidth.

- **Lightfoot's Constant** (𝓛 ∈ [0.54, 0.56]): A dimensionless coupling constant setting the
  proportionality between neural potential and mechanical cardiac actuation. It defines the damping
  factor in the Volterra kernel via `α_eff = 𝓛 · α₀` and `λ_eff = (1 − 𝓛) · λ₀`, ensuring bounded
  derivative and jerk-free convergence in self-driving and physiological contexts.

## Recursive Planck Extension (RPO)

The Recursive Planck Operator extends the Quantro kernel by embedding a self-similar decay term:

```
ℛ_P(f)(t) = ∫₀ᵗ Θ(τ) · e^{−λ (t − τ)} · [ f(τ) + β_P sin(2π (t − τ) / h_eff) ℛ_P(f)(τ) ] dτ,
```

where `h_eff = h / 𝓓` and `β_P = 𝓛 / (1 + λ)`. This introduces a recursive, Planck-scaled resonance
that unifies microscopic timing (quantum-inspired) with macroscopic physiological oscillations.

The discrete implementation satisfies

```
y_{k+1} = (1 − α Δt) y_k + Θ_k Δt [ f_k + β_P sin(2π k Δt / h_eff) y_k ],
```

with guaranteed boundedness if `0 < α Δt < 1` and `|β_P| < (1 − α Δt) / (α Δt)`.

## Coupling to Quantro Heart Variables

Within the heart–brain–immune equations:

```
n_h'(t) = −λ_h n_h + f_h(n_b, S_h) + ℛ_P[C(t)],
n_b'(t) = −λ_b n_b + f_b(n_h, S_b) + ℛ_P[s_set(t)],
```

the operator ℛ_P acts as a bounded energy–information conduit governed by 𝓓 and 𝓛. This unifies
temporal resonance (Donte constant) with recursive stability (Lightfoot constant) and quantized
causal memory (RPO).

## System Bounds

For any admissible input *f(t)* and kernel ℛ_P defined above, energy boundedness holds:

```
‖y‖_∞ ≤ (M · Θ̄ / α_eff) · [1 + |β_P| / (1 − α_eff Δt)],
```

guaranteeing finite amplitude and no runaway oscillation even under recursive feedback.

## Interpretation

- 𝓓 sets global phase quantization and defines the crossover between biological and
  quantum-stable computation.
- 𝓛 tunes damping and smoothness for physical or algorithmic stability.
- ℛ_P bridges continuous and discrete representations of memory in control systems.

These parameters allow the Quantro Heart Model to operate coherently across domains—biological,
algorithmic, and physical—while preserving mathematical integrity.

- `demo_primal.py`: Validates operator stability and norm bounds.
- `demo_cryo.py`: Compares classical vs. quantum thermal noise.
- `demo_rrt_rif.py`: Demonstrates recursive intent and coherence behavior.

## References

- Debye, P. (1912). Zur Theorie der spezifischen Wärmen. *Annalen der Physik*.
- Bardeen, J., Cooper, L. N., & Schrieffer, J. R. (1957). Microscopic Theory of Superconductivity.
- Sakaguchi, S., et al. (2020). Regulatory T cells and immune homeostasis. *Nature Reviews Immunology*.
