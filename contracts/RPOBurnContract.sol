// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * RPO Token Burn Contract for PrimalRWA Integration
 *
 * This contract handles token burns for MotorHandPro actuation.
 * Exchange rate: 1 RPO token = 1 second of robotic actuation
 *
 * Patent Pending: U.S. Provisional Patent Application No. 63/842,846
 * Copyright 2025 Donte Lightfoot - The Phoney Express LLC
 */

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract RPOBurnContract is Ownable, ReentrancyGuard {

    IERC20 public rpoToken;

    // Mapping of authorized actuators to their blockchain addresses
    mapping(address => bool) public authorizedActuators;

    // Burn tracking per actuator
    mapping(address => uint256) public totalBurnedByActuator;

    // Total tokens burned across all actuators
    uint256 public totalBurned;

    // Events
    event TokensBurned(
        address indexed actuator,
        uint256 seconds,
        uint256 tokensBurned,
        uint256 timestamp
    );

    event ActuatorAuthorized(address indexed actuator);
    event ActuatorRevoked(address indexed actuator);

    /**
     * @dev Constructor
     * @param _rpoToken Address of the RPO token contract
     */
    constructor(address _rpoToken) {
        require(_rpoToken != address(0), "Invalid token address");
        rpoToken = IERC20(_rpoToken);
    }

    /**
     * @dev Authorize an actuator to burn tokens
     * @param actuator Address of the actuator to authorize
     */
    function authorizeActuator(address actuator) external onlyOwner {
        require(actuator != address(0), "Invalid actuator address");
        authorizedActuators[actuator] = true;
        emit ActuatorAuthorized(actuator);
    }

    /**
     * @dev Revoke an actuator's authorization
     * @param actuator Address of the actuator to revoke
     */
    function revokeActuator(address actuator) external onlyOwner {
        authorizedActuators[actuator] = false;
        emit ActuatorRevoked(actuator);
    }

    /**
     * @dev Burn tokens for actuation time
     * @param actuator Address of the actuator burning tokens
     * @param seconds Number of seconds of actuation (1 token per second)
     *
     * Exchange Rate: 1 RPO Token = 1 Second of Actuation
     */
    function burnTokens(address actuator, uint256 seconds) external nonReentrant {
        require(authorizedActuators[actuator], "Actuator not authorized");
        require(seconds > 0, "Must burn for at least 1 second");

        // Calculate tokens to burn (1 token = 1 second)
        uint256 tokensToBurn = seconds * 1e18; // Assuming 18 decimals

        // Transfer tokens from treasury to this contract
        require(
            rpoToken.transferFrom(msg.sender, address(this), tokensToBurn),
            "Token transfer failed"
        );

        // Burn the tokens (send to 0x0 address)
        require(
            rpoToken.transfer(address(0), tokensToBurn),
            "Token burn failed"
        );

        // Update tracking
        totalBurnedByActuator[actuator] += seconds;
        totalBurned += tokensToBurn;

        // Emit event
        emit TokensBurned(actuator, seconds, tokensToBurn, block.timestamp);
    }

    /**
     * @dev Get total seconds burned by an actuator
     * @param actuator Address of the actuator
     * @return Total seconds burned
     */
    function getBurnedSeconds(address actuator) external view returns (uint256) {
        return totalBurnedByActuator[actuator];
    }

    /**
     * @dev Get total tokens burned (in wei)
     * @return Total tokens burned
     */
    function getTotalBurned() external view returns (uint256) {
        return totalBurned;
    }
}
