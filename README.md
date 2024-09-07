# DLC_FST_Analysis

This repository contains Python scripts for analyzing rodent behavior during Forced Swim Test (FST) using DeepLabCut (DLC). The analysis is conducted using custom Python scripts designed to process DLC-generated CSV files and human assessment data.

## Repository Structure

- **`calibration.py`**: Script to fine-tune the "sense" threshold, which determines whether a rodent is classified as 'moving' or 'not moving' based on the distance between body part coordinates in consecutive frames. The optimized `sense` value is obtained by maximizing the correlation between automated analysis and manual human evaluations using different optimization methods.
- **`analyzer.py`**: Script that uses the optimized `sense` value from `calibration.py` to analyze DLC output data. It reads the DLC CSV files, applies the threshold to classify movements, and outputs movement metrics into a CSV file.
- **`requirements.txt`**: Contains the Python libraries required to run the scripts.

## Installation

To run the scripts, you need to have Python installed along with the required libraries. You can install the necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
