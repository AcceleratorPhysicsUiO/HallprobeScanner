# Hall-Probe Magnetic-Field Measurement Using a THM1176-MF

**Date:** 1-12 October 2026 (ongoing)  
**Script:** `StandaTrans_live.py`

## Overview

This project measures magnetic-field distributions using a Metrolab THM1176-MF three-axis Hall probe mounted on a motorised Standa translation stage. The Python script automates stage movement, data acquisition, live plotting, and CSV export.

## Hardware and software

- **Hall probe:** Metrolab THM1176-MF
- **Translation stage:** Standa, controlled via `libximc`
- **Python libraries:** NumPy, Matplotlib, `pyTHM1176`, `usbtmc`, and `libximc`

## Measurement procedure

The script scans the stage through predefined positions, acquires \(B_x\), \(B_y\), and \(B_z\), and displays the measurements live. Only samples acquired while the stage is stationary at each target position are retained for the final dataset.

Default scan settings:
- Start: 0 mm
- End: 250 mm
- Step size: 2 mm
- Dwell time: 1 s per position
- Hall-probe range: 0.1 T

The stage calibration is 500 mm per 25,000 steps, corresponding to 20 µm per step.

## Data output

Measurements are saved as timestamped CSV files containing the magnetic-field components, probe temperature, stage position in metres, and motor position in steps. A final plot shows the field components against stage position.

**Important:** The CSV field-component ordering is intentional. The script writes the probe channels in the order `By`, `Bz`, `Bx` under the headers `Bx_T`, `By_T`, `Bz_T`, respectively, following the convention required by the Hall-probe field-map workflow. 
Do not reorder these columns without updating the corresponding analysis.

## Purpose

The measurements support experimental characterisation of the magnetic field maps of magnetic elements.
