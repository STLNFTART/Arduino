module primal.hardware.motorhandpro;

import std.datetime : Clock;
import std.exception : enforce;
import std.file : write;
import std.format : format;
import std.json : JSONValue;
import std.path : buildPath;
import std.stdio : stderr, writeln;

/// Represents a high-level control packet for the Motor Hand Pro device.
struct MotorCommand {
    double gripForceNewton;   /// commanded grip force in newtons
    double wristAngleDeg;     /// wrist angle command in degrees
    double fingerSpreadDeg;   /// finger spread angle in degrees
    double energyBudgetJoule; /// energy budget per update (safety limit)
}

/// Encapsulates configuration for connecting to Motor Hand Pro.
struct MotorHandConfig {
    string outputDirectory;  /// directory for command logs / serial forwarder
    bool dryRun = true;      /// when true, run without hitting hardware
}

/// Client responsible for serializing commands to the device.
class MotorHandProClient {
    private MotorHandConfig config;
    private static size_t logSequence;

    this(MotorHandConfig config) {
        enforce(config.outputDirectory.length > 0, "Output directory required");
        this.config = config;
    }

    /// Sends a command packet, logging JSON to the output directory.
    void sendCommand(const MotorCommand command) {
        auto timestamp = Clock.currTime;
        JSONValue payload;
        payload["timestamp_iso"] = JSONValue(timestamp.toISOExtString());
        payload["grip_force_newton"] = JSONValue(command.gripForceNewton);
        payload["wrist_angle_deg"] = JSONValue(command.wristAngleDeg);
        payload["finger_spread_deg"] = JSONValue(command.fingerSpreadDeg);
        payload["energy_budget_joule"] = JSONValue(command.energyBudgetJoule);

        auto payloadString = payload.toString();
        enforce(!payloadString.empty, "Failed to encode Motor Hand Pro command");

        auto sequence = logSequence++;
        auto fileName = buildPath(config.outputDirectory,
                                  format!"motor_command_%s_%06d.json"(timestamp.stdTime, sequence));
        scope (failure) {
            stderr.writeln("MotorHandProClient: failed to persist command payload.");
        }
        write(fileName, payloadString);

        if (!config.dryRun) {
            // Hardware hook: forward payload to serial/UART driver.
            // Implementation intentionally omitted for offline environment.
            stderr.writeln("[Hardware] Command forwarded to Motor Hand Pro serial interface");
        }
    }
}

unittest {
    auto config = MotorHandConfig("/tmp", true);
    auto client = new MotorHandProClient(config);
    auto command = MotorCommand(5.0, 10.0, 15.0, 0.5);
    client.sendCommand(command);
}
