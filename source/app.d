module app;

import std.algorithm.comparison : clamp;
import std.file : mkdirRecurse, write, append;
import std.getopt : getopt, defaultGetoptPrinter;
import std.path : buildPath;
import std.stdio : writeln;
import std.string : format;

import primal.config;
import primal.quantum;
import primal.plasma;
import primal.coherence;
import primal.control;
import primal.hardware.motorhandpro;
import primal.simulation.cryosim;

/// Application entry point
int main(string[] args) {
    size_t steps = 200;
    string outputDir = buildPath("output", "motor_hand_logs");
    bool hardware = false;
    size_t gridSize = 8;

    auto helpInformation = getopt(args,
        "steps", &steps,
        "output-dir", &outputDir,
        "hardware", &hardware,
        "grid", &gridSize);

    if (helpInformation.helpWanted) {
        defaultGetoptPrinter("PRIMAL LOGIC Framework v1.0.0", helpInformation.options); 
        return 0;
    }

    mkdirRecurse(outputDir);

    auto params = PrimalParameters();
    auto state = makeQuantumState(gridSize, gridSize);
    PlasmaField plasma;
    plasma.values = new double[](gridSize);
    plasma.spacingMm = spatialScalesMm[1];
    auto cryoState = CryoState(260.0, 0.15, 0.25);
    auto organs = defaultOrgans(params);

    double collapseProb = 0;
    auto motorClient = new MotorHandProClient(MotorHandConfig(outputDir, !hardware));

    auto csvPath = buildPath(outputDir, "simulation_log.csv");
    write(csvPath, "step,time_ms,alpha,plasma_energy,vitality,collapse_prob,grip_force\n");

    foreach (step; 0 .. steps) {
        double timeMs = step * 1.0;
        auto mix = computeSuperposition(params.alpha, timeMs);

        auto nextState = evolveQuantumState(state, params, /*gammaField*/0.2,
                                            mix[0], mix[1],
                                            0.001, 1.0, /*plasmaGamma*/0.1,
                                            collapseProb);

        updatePlasmaField(plasma, params, 0.001);
        auto energy = plasmaEnergy(plasma);

        double[][] coherenceVectors;
        coherenceVectors.length = 3;
        foreach (i; 0 .. coherenceVectors.length) {
            auto rowIndex = (i * 2) % gridSize;
            coherenceVectors[i] = nextState.realPart[rowIndex].dup;
        }
        auto coherence = coherenceScore(coherenceVectors);

        auto alpha = adaptiveAlpha(params, params.alpha, energy, coherence, timeMs);

        auto updatedCryo = updateCryoState(cryoState, organs, params, 0.001, mix[0], mix[1]);
        cryoState = updatedCryo;

        auto gripForce = clamp(updatedCryo.vitality * 100.0, 0.0, 120.0);
        auto command = MotorCommand(gripForce, 10.0 + 5.0 * coherence, 15.0, params.energyBudget);
        motorClient.sendCommand(command);

        auto line = format("%s,%s,%.4f,%.6f,%.4f,%.4f,%.2f\n",
                           step, timeMs, alpha, energy, cryoState.vitality, collapseProb, gripForce);
        append(csvPath, line);

        state = nextState;
    }

    writeln("Simulation complete. Logs stored in ", outputDir);
    return 0;
}
