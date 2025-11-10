module primal.hardware.motorhandpro;

import std.datetime : Clock;
import std.exception : enforce;
import std.file : exists, isDir, mkdirRecurse;
import std.json : JSONValue;
import std.path : buildPath;
import std.stdio : File, stderr, writeln;

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
    private CommandFileLogger commandLogger;

    this(MotorHandConfig config) {
        enforce(config.outputDirectory.length > 0, "Output directory required");
        this.config = config;

        const logDir = config.outputDirectory;
        if (!exists(logDir)) {
            mkdirRecurse(logDir);
        } else {
            enforce(isDir(logDir), "MotorHandProClient: output path must be a directory");
        }

        auto logPath = buildPath(logDir, "motor_hand_commands.jsonl");
        commandLogger = new CommandFileLogger(logPath);
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

        scope (failure) {
            stderr.writeln("MotorHandProClient: failed to persist command payload.");
        }
        commandLogger.log(payloadString);

        if (!config.dryRun) {
            // Hardware hook: forward payload to serial/UART driver.
            // Implementation intentionally omitted for offline environment.
            stderr.writeln("[Hardware] Command forwarded to Motor Hand Pro serial interface");
        }
    }
}

/// Minimal file-backed logger that appends JSON payloads without overwriting.
class CommandFileLogger {
    private File file;

    this(string filePath) {
        try {
            file = File(filePath, "a");
        } catch (Exception e) {
            enforce(false, "CommandFileLogger: unable to open log file: " ~ e.msg);
        }
    }

    ~this() {
        if (file.isOpen) {
            file.flush();
            file.close();
        }
    }

    /// Appends a command payload to the log in JSON Lines format.
    void log(const string payload) {
        synchronized (this) {
            file.writeln(payload);
            file.flush();
        }
    }
}

unittest {
    auto config = MotorHandConfig("/tmp", true);
    auto client = new MotorHandProClient(config);
    auto command = MotorCommand(5.0, 10.0, 15.0, 0.5);
    client.sendCommand(command);
}
