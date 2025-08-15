# Copilot Instructions for EE400 Final Project

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

This is an Arduino project for an EE400 (Electrical Engineering) Final Project that implements computer vision and machine learning on an Arduino Nano 33 BLE Sense microcontroller.

## Project Context

- **Hardware**: Arduino Nano 33 BLE Sense with OV7675 camera module
- **Framework**: Arduino IDE/Framework with Edge Impulse machine learning library
- **Purpose**: Person detection using computer vision and machine learning inference
- **Language**: C++ (Arduino variant)

## Key Components

- Edge Impulse machine learning inference library (`FinalProject3_inferencing.h`)
- Arduino OV767X camera library for image capture
- Image processing and classification for person detection
- Serial communication for debugging and data output

## Coding Guidelines

- Follow Arduino C++ conventions and best practices
- Use appropriate data types for embedded systems (consider memory constraints)
- Include proper error handling for camera operations and inference
- Maintain clear separation between camera operations, inference, and communication
- Use descriptive variable names and comments for complex vision processing code
- Consider power efficiency and real-time performance requirements

## Libraries and Dependencies

- `FinalProject3_inferencing.h` - Edge Impulse generated library
- `Arduino_OV767X.h` - Camera module library
- Standard Arduino libraries (Serial, etc.)

When suggesting code improvements or modifications, keep in mind the memory and processing constraints of the Arduino Nano 33 BLE Sense platform.
