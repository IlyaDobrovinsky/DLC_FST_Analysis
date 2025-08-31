import pandas as pd
import numpy as np
import os
import glob
import logging
from scipy.optimize import minimize, minimize_scalar

# Logging setup
logging.basicConfig(filename="processing_errors.log", level=logging.ERROR)

# ------------------------
# Helpers
# ------------------------
def get_csv_files_from_folder(folder_name):
    return glob.glob(os.path.join(folder_name, "*.csv"))

def extract_info_from_filename(file_name):
    parts = file_name.split('_')
    name = parts[0]
    gender = 'Male' if parts[1] == 'M' else 'Female'
    group = 'SI' if name[1] == '1' else ('Control' if name[1] == '2' else 'FWD')
    day = 'Test' if '_TESTDLC' in file_name else ('Habituation' if '_HABIDLC' in file_name else 'Unknown')
    return name, gender, group, day

# ------------------------
# Calibration
# ------------------------
def read_matching_files(human_raters_file="Human_raters_means.xlsx", folder="test"):
    human_raters_df = pd.read_excel(human_raters_file) 
    vid_list = human_raters_df['vid'].tolist()
    csv_data_dict = {}
    for vid in vid_list:
        csv_file_name = os.path.join(folder, vid + 'DLC_resnet50_FSTshuffle1_500000.csv')
        if os.path.exists(csv_file_name):
            csv_data_dict[vid] = pd.read_csv(csv_file_name)
    return csv_data_dict, human_raters_df

def negative_correlation(sense, csv_data_dict, human_raters_df):
    moving_values = []
    for vid, df in csv_data_dict.items():
        df_temp = df.iloc[:, 18:21].apply(pd.to_numeric, errors='coerce')
        body_center = df_temp[df_temp[df_temp.columns[2]] > 0.95]
        distances = np.sqrt(body_center.iloc[:, 0].diff() ** 2 + body_center.iloc[:, 1].diff() ** 2)
        movements = np.where(distances > sense, 'moving', 'not moving')
        movement_counts = np.bincount(movements == 'moving') / 30
        moving_values.append(movement_counts[1])
    correlation_value = np.corrcoef(moving_values, human_raters_df['moving'].tolist())[0, 1]
    return -correlation_value

def calibrate_sense(human_raters_file="Human_raters_means.xlsx", folder="test"):
    csv_data_dict, human_raters_df = read_matching_files(human_raters_file, folder)
    best_overall_correlation = -float('inf')
    best_overall_sense, best_method = None, None

    for method in ['brent']:
        result = minimize_scalar(lambda s: negative_correlation(s, csv_data_dict, human_raters_df),
                                 bracket=(0, 1), method=method)
        if -result.fun > best_overall_correlation:
            best_overall_correlation, best_overall_sense, best_method = -result.fun, result.x, method

    for method in ['Nelder-Mead', 'L-BFGS-B']:
        result = minimize(lambda x: negative_correlation(x[0], csv_data_dict, human_raters_df),
                          [0.5], bounds=[(0, 1)], method=method)
        if -result.fun > best_overall_correlation:
            best_overall_correlation, best_overall_sense, best_method = -result.fun, result.x[0], method

    return best_overall_sense, best_overall_correlation, best_method

# ------------------------
# Analysis
# ------------------------
def analyze_movement(file_path, sense):
    try:
        cols_to_read = list(range(18, 21))  # body center
        df = pd.read_csv(file_path, skiprows=1, usecols=cols_to_read, low_memory=False)
        df = df.apply(pd.to_numeric, errors='coerce')
        body_center = df.copy()
        body_center.columns = ['x', 'y', 'likelihood']

        total_frames = len(body_center)
        body_center = body_center[body_center['likelihood'] > 0.95]
        likelihood_above_95_percentage = (len(body_center) / total_frames) * 100

        body_center['distance'] = np.sqrt(body_center['x'].diff() ** 2 + body_center['y'].diff() ** 2)
        body_center['movement'] = np.where(body_center['distance'] > sense, 'moving', 'not moving')

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

def analyze_folder(folder="test", sense=0.48, output_filename="FST_results.csv"):
    csv_files = get_csv_files_from_folder(folder)
    results = [analyze_movement(file, sense) for file in csv_files if analyze_movement(file, sense) is not None]
    if results:
        results_df = pd.concat(results)
        results_df.to_csv(output_filename, index=False)
        return results_df
    return None

# ------------------------
# Main pipeline
# ------------------------
def main():
    folder = "test"
    best_sense, best_corr, method = calibrate_sense(folder=folder)
    print(f"Optimal sense: {best_sense:.4f}, correlation={best_corr:.3f}, method={method}")
    results_df = analyze_folder(folder=folder, sense=best_sense)
    if results_df is not None:
        print("Results saved successfully.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
