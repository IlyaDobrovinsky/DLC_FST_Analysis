import pandas as pd
import numpy as np
import os
import glob
import logging
from scipy.optimize import minimize, minimize_scalar

# Setting up logging for error tracking
logging.basicConfig(filename="processing_errors.log", level=logging.ERROR)

csv_data_dict = None
human_raters_df = None


def read_matching_files():
    human_raters_df = pd.read_excel("Human_raters_means.xlsx") 
    vid_list = human_raters_df['vid'].tolist()

    # Dictionary to store dataframes of matched CSV files
    csv_data_dict = {}

    # Load matched CSV files into dictionary
    for vid in vid_list:
        csv_file_name = os.path.join('test', vid + 'DLC_resnet50_FSTshuffle1_500000.csv')
        if os.path.exists(csv_file_name):
            csv_data_dict[vid] = pd.read_csv(csv_file_name)

    return csv_data_dict, human_raters_df


def negative_correlation(sense):
    global csv_data_dict, human_raters_df

    moving_values = []
    for vid, df in csv_data_dict.items():
        df_temp = df.iloc[:, 6 * 3:6 * 3 + 3].apply(pd.to_numeric, errors='coerce')
        body_center = df_temp[df_temp[df_temp.columns[2]] > 0.95]
        distances = np.sqrt(body_center.iloc[:, 0].diff() ** 2 + body_center.iloc[:, 1].diff() ** 2)
        movements = np.where(distances > sense, 'moving', 'not moving')
        movement_counts = np.bincount(movements == 'moving') / 30
        moving_values.append(movement_counts[1])

    correlation_value = np.corrcoef(moving_values, human_raters_df['moving'].tolist())[0, 1]
    return -correlation_value


def sense_fine_tune():
    global csv_data_dict, human_raters_df
    csv_data_dict, human_raters_df = read_matching_files()

    methods_scalar = ['brent']
    methods_vector = ['Nelder-Mead', 'L-BFGS-B']

    best_overall_correlation = -float('inf')
    best_overall_sense = None
    best_method = None

    for method in methods_scalar:
        result = minimize_scalar(negative_correlation, bracket=(0, 1), method=method)
        optimal_sense = result.x
        best_correlation = -result.fun
        if best_correlation > best_overall_correlation:
            best_overall_correlation = best_correlation
            best_overall_sense = optimal_sense
            best_method = method

    for method in methods_vector:
        result = minimize(lambda x: negative_correlation(x[0]), [0.5], bounds=[(0, 1)], method=method)
        optimal_sense = result.x[0]
        best_correlation = -result.fun
        if best_correlation > best_overall_correlation:
            best_overall_correlation = best_correlation
            best_overall_sense = optimal_sense
            best_method = method

    return best_overall_sense, best_overall_correlation, best_method


def get_csv_files_from_folder(folder_name):
    return glob.glob(os.path.join(folder_name, "*.csv"))


def extract_info_from_filename(file_name):
    parts = file_name.split('_')
    name = parts[0]
    gender = 'Male' if parts[1] == 'M' else 'Female'
    group = 'SI' if name[1] == '1' else ('Control' if name[1] == '2' else 'FWD')
    day = 'Test' if '_TESTDLC' in file_name else ('Habituation' if '_HABIDLC' in file_name else 'Unknown')
    return name, gender, group, day


def analyze_movement(file_path, sense):
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


def analyze_files_in_folder(folder):
    csv_files = get_csv_files_from_folder(folder)
    sense = sense_fine_tune()
    results = [analyze_movement(file, sense) for file in csv_files if analyze_movement(file, sense) is not None]
    return results


def save_results_to_csv(results, output_filename='FST_results_GENNAME.csv'): #FST_results
    if results:
        results_df = pd.concat(results)
        results_df.to_csv(output_filename, index=False)


def main():
    folder = 'test'
    results = analyze_files_in_folder(folder)
    save_results_to_csv(results)
    best_sense, best_correlation, best_method = sense_fine_tune()
    print(f"Best sense value: {best_sense}")
    print(f"Best correlation: {best_correlation}")
    print(f"Best method: {best_method}")


if __name__ == "__main__":
    main()
