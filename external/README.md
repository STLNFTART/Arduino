# External Integrations

This directory contains Git submodules and integration glue for third-party components.

- `MotorHandPro/` – mirror of https://github.com/STLNFTART/MotorHandPro used for cross-repo
  development and hardware bridge coordination. Run the following to fetch it:

  ```bash
  git submodule update --init --recursive
  ```

The submodule checkout is optional for pure simulation workflows but required when testing the
Motor Hand Pro hardware bridge implementation.
