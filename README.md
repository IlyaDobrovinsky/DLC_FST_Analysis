# DLC_FST_Analysis

This repository contains a unified Python script for analyzing rodent behavior during the Forced Swim Test (FST) using DeepLabCut output and calibration against human raters.

## Features
- **Calibration**: Optimizes the "sense" threshold by maximizing correlation between automated classification and human ratings.
- **Analysis**: Applies the optimized threshold to classify movements as "moving" or "not moving" across all DLC CSV files.
- **Output**: Produces a CSV file with movement metrics for each subject.

## Installation
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
