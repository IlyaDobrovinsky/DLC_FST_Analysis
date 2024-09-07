import pandas as pd
import numpy as np
import os
import glob
import logging

# Setting up logging for error tracking
logging.basicConfig(filename="processing_errors.log", level=logging.ERROR)

# Using the sense value from the output as a constant
SENSE_VALUE = 0.48750000000000004 # Change value bases on Calibration.py output.

def get_csv_files_from_folder(folder_name):
    return glob.glob(os.path.join(folder_name, "*.csv"))

def extract_info_from_filename(file_name):
    parts = file_name.split('_')
    name = parts[0]
    gender = 'Male' if parts[1] == 'M' else 'Female'
    group = 'SI' if name[1] == '1' else ('Control' if name[1] == '2' else 'FWD')
    day = 'Test' if '_TESTDLC' in file_name else ('Habituation' if '_HABIDLC' in file_name else 'Unknown')
    return name, gender, group, day

def analyze_movement(file_path):
    try:
        cols_to_read = list(range(6 * 3, 6 * 3 + 3))
        df = pd.read_csv(file_path, skiprows=1, usecols=cols_to_read, low_memory=False)
        df = df.apply(pd.to_numeric, errors='coerce')
        body_center = df.copy()
        body_center.columns = ['x', 'y', 'likelihood']
        total_frames = len(body_center)
        body_center = body_center[body_center['likelihood'] > 0.95]
        likelihood_above_95_percentage = (len(body_center) / total_frames) * 100
        body_center['distance'] = np.sqrt(body_center['x'].diff() ** 2 + body_center['y'].diff() ** 2)
        body_center['movement'] = np.where(body_center['distance'] > SENSE_VALUE, 'moving', 'not moving')
        movement_counts = body_center['movement'].value_counts() / 30
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        name, gender, group, day = extract_info_from_filename(file_name)
        return pd.DataFrame({
            'name': [name],
            'gender': [gender],
            'group': [group],
            'day': [day],
            'file_name': [file_name],
            'moving': [movement_counts.get('moving', 0)],
            'not moving': [movement_counts.get('not moving', 0)],
            'total_time': [movement_counts.get('moving', 0) + movement_counts.get('not moving', 0)],
            'likelihood_above_95_percentage': [likelihood_above_95_percentage]
        })
    except Exception as e:
        logging.error(f"Failed to process file: {file_path}. Error: {e}")
        return None

def analyze_files_in_folder(folder):
    csv_files = get_csv_files_from_folder(folder)
    results = [analyze_movement(file) for file in csv_files if analyze_movement(file) is not None]
    return results

def save_results_to_csv(results, output_filename='GenNAME_results.csv'):
    if results:
        results_df = pd.concat(results)
        results_df.to_csv(output_filename, index=False)

def main():
    folder = 'test'
    results = analyze_files_in_folder(folder)
    save_results_to_csv(results)

if __name__ == "__main__":
    main()
