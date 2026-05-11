# E201 Test Platform

Python-based GUI application for testing, debugging and analysing rotary encoders using any encoder interfaces and a reference encoder system.

The application supports:
- Live encoder position monitoring
- Error / noise analysis
- High-speed synchronized acquisition
- Plot recording and export
- Register access
- Encoder configuration
- Motor control
- Calibration and diagnostics tools


Supports incremental, SSI, BiSS and SPI encoder communication depending on connected E201 hardware.

---

## Supported Hardware

### E201 Interfaces
- E201-9B (BiSS-C bidirectional)
- E201-9Q (Incremental)
- E201-9S (SSI / BiSS unidirectional)
- E201-9P (SPI / PWM)

### Supported Communication Types
- Incremental ABZ
- SSI
- BiSS-C
- SPI

### Motor Controllers
- EPOS

---

# Main Features

### Live Position Reading
Continuous synchronized acquisition of:
- DUT encoder
- Reference encoder

Displayed values:
- DUT counts
- REF counts
- DUT degrees
- REF degrees
- Multiturn value

Supports:
- Constant position reading
- Triggered synchronized reading
- Inverted DUT direction
- Adjustable plot buffer size

---

## Position Plot

Real-time plotting using PyQtGraph.

### Supported Views
- DUT
- REF
- DUT + REF

### Supported Units
- Degrees
- Counts

---

## Analysis Plot

Real-time analysis between DUT and reference encoder.

### Analysis Modes
- Error
- Noise
- INL
- DNL

### Metrics
Displayed live:
- Error P2P
- Error RMS
- Noise P2P
- Standard deviation

#3# Zero Offset
`ZERO OFFSET` removes static offset between DUT and reference.

---

## Data Recording

Supports:
- Continuous acquisition recording
- Fixed sample recording

Recorded data includes:
- Sample index
- Timestamp
- DUT counts
- REF counts
- DUT scaled values
- REF scaled values

### Save Plot
Exports:
- Plot image
- CSV measurement data

Useful for:
- Offline analysis
- MATLAB/Python processing
- Production testing logs

---

## DUT Connection

### Supported Settings
Depending on communication type:
- Resolution
- Singleturn bits
- Multiturn bits
- Status bits
- CRC bits
- DUT bytes
- SPI polarity
- SPI phase
- SPI frequency

### Power Supply
Supports:
- 3.3 V
- 5 V

### Communication Setup
Communication can be configured directly from GUI:
- ABZ
- SSI
- BiSS
- SPI

---

## Reference Encoder Connection

Separate reference encoder interface:
- Independent COM selection
- Independent E201 selection

Typically used with:
- High precision optical encoder

---

## Register Access

Low-level encoder register communication.

### Features
- Read registers
- Write registers
- Signed/unsigned values
- Variable register lengths
- Variable banks

### Predefined Register Sets
Examples:
- AksIM BiSS
- SPI encoder registers

Useful for:
- Diagnostics
- Encoder configuration
- Development

---

## Encoder Configuration Tools

GUI provides direct encoder configuration commands:

### Supported Actions
- Multiturn set
- Position offset set
- Counting direction change
- Factory reset
- Save to flash
- Start calibration

---

### MIS Image Readout

Supports reading and visualization of:
- MIS image data
- Raw magnetic signal diagnostics

Useful for:
- Magnet alignment verification
- Installation diagnostics
- Airgap analysis

---



## Motor Control

Integrated motor control module used for automated encoder testing and repeatable measurements.

Currently supported:
- EPOS motor controllers

The architecture is designed to allow easy integration of additional motor drivers in the future.

## Features
- Motor enable
- Speed control
- Live RPM display
- Stop command

Allows:
- Automated encoder rotation
- Repeatable measurements
- Noise/error characterization during movement

---

## High-Speed Acquisition Architecture

The GUI uses:
- Dedicated acquisition thread
- Circular ring buffers
- Real-time plotting

Typical operation:
- ~1 kHz acquisition loop
- ~30 FPS GUI refresh

Designed for:
- Low-latency acquisition
- Stable long-duration measurements
- Continuous recording

---

# Installation

## Requirements
- Python 3.11
- Windows

## Install uv

```bash
pip install uv
```

## Run GUI

```bash
_run_Gui_python311.bat
```

---

# Typical Workflow

## 1. Connect Devices
- Select DUT COM port
- Select REF COM port
- Click `CONNECT`

## 2. Configure Parser
Set:
- Communication type
- Resolution
- Bit structure
- SPI settings if applicable

## 3. Start Reading
Enable:
- Constant position reading

## 4. Analyse
Choose:
- Error
- Noise
- INL
- DNL

## 5. Record Data
- Start recording
- Save plot and CSV

---

# Intended Use

The platform is intended for:
- Encoder validation
- Noise characterization
- Mechanical testing
- Production diagnostics
- Calibration verification
- Research and development
