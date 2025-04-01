import pandas as pd
import numpy as np
import os

def get_prob_0(df, name):
    # Check if the 'class_1_proba' column exists
    if 'y_prob_pool_average' in df.columns:
        # Calculate the class 0 probabilities
        df['y_prob_pool_0'] = 1 - df['y_prob_pool_average']

        # Save the updated DataFrame to a new CSV file
        df.to_csv(os.path.join(name,'y_prob_pool_average_cnn_binary.csv'), index=True)
        print("The updated CSV file 'y_prob_pool_average_cnn_binary.csv' has been created successfully.")
    else:
        print("The column 'y_prob_pool_average' does not exist in the dataset.")

def uncertainty(name, df):
    df["most_likely_prob"] = df.max(axis=1) 
    df["uncertain_prob"] = 1 - df["most_likely_prob"]
    df = df.drop(columns=["most_likely_prob"])
    df.to_csv(os.path.join(name, "uncertain_prob.csv"), index=True)
    print("Uncertainty calculation for proba finished.")

def uncertain_sort(name, df):
    df["distance_to_0.5"] = abs(df["uncertain_prob"] - 0.5) # Compute how close each uncertain_prob is to 0.5
    df = df.sort_values(by="distance_to_0.5")   # Sort by how close values are to 0.5 (smallest first)
    n = int(len(df) * 0.05)  # Select the top n% of rows
    uncertain_subset = df.iloc[:n].drop(columns=["distance_to_0.5"])  # Drop helper column
    remaining_data = df.iloc[n:].drop(columns=["distance_to_0.5"])

    # Save the datasets
    uncertain_subset.to_csv(os.path.join(name, "uncertain_subset.csv"), index=True)
    remaining_data.to_csv(os.path.join(name, "remaining_pool.csv"), index=True)

    print(f"Split completed: {n} rows saved in 'uncertain_subset.csv'.")

def split_y_pool(name, df, uncertain_subset_df):
    uncertain_y_pool = df[df["LigandID"].isin(uncertain_subset_df["LigandID"])]
    remaining_y_pool = df[~df["LigandID"].isin(uncertain_subset_df["LigandID"])]
    uncertain_y_pool.to_csv(os.path.join(name, "uncertain_subset_y_pool.csv"), index=False)
    remaining_y_pool.to_csv(os.path.join(name, "remaining_y_pool.csv"), index=False)
    print("y_pool split subset and remaining.")

def split_data(large_filepath, large_filename, list_filepath, list_filename, output_path, filtered_list,remaining_list):
    large_df = pd.read_csv(os.path.join(large_filepath, large_filename))        # Load the large data
    compound_list_df = pd.read_csv(os.path.join(list_filepath, list_filename))  # Load the compound list

    # Check if 'LigandID' columns are in both DataFrames
    if 'LigandID' not in large_df.columns:
        print(f"'LigandID' column not found in {large_filename}")
    if 'LigandID' not in compound_list_df.columns:
        print(f"'LigandID' column not found in {list_filename}")
    
    large_df['LigandID'] = large_df['LigandID'].astype(str).str.strip()
    compound_list_df['LigandID'] = compound_list_df['LigandID'].astype(str).str.strip()
    filtered_list_df = large_df[large_df['LigandID'].isin(compound_list_df['LigandID'])]
    # Save the filtered DataFrame to a new CSV file
    filtered_list_df.to_csv(os.path.join(output_path, filtered_list), index=False)

    print("CSV files split generated successfully.")


def merge_dataframes(file_paths, output_path, how='outer'):
    # Read and merge the DataFrames
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    df_merged = pd.concat(dfs, axis=0, ignore_index=True, join=how)
    
    df_merged.to_csv(output_path, index=False)      # Save the merged DataFrame to CSV

def main():
    name = "uncertain1"
    path_file = 'pool_pred/pool_1337'
    df = pd.read_csv(os.path.join(name, 'meta_pool_cnn', 'y_prob_pool_average_cnn.csv'), index_col=0)
    get_prob_0(df, name)
    df = pd.read_csv(os.path.join(name, "y_prob_pool_average_cnn_binary.csv"), index_col=0)
    uncertainty(name, df)
    uncertain_cal_file = pd.read_csv(os.path.join(name, "uncertain_prob.csv"), index_col=0)
    uncertain_sort(name, uncertain_cal_file)
    y_pool = pd.read_csv(os.path.join(path_file,"y_pool.csv"))
    uncertain_subset_df = pd.read_csv(os.path.join(name, "uncertain_subset.csv"))
    split_y_pool(name, y_pool, uncertain_subset_df)

    large_filepath = 'data'
    large_filename = 'tpo_combine_process.csv'
    list_filepath = name
    list_filename = 'uncertain_subset.csv'
    output_path = name
    filtered_list = 'x_subset_0.05.csv'
    remaining_list = 'remaining_trainset.csv'
    split_data(large_filepath, large_filename, list_filepath, list_filename, output_path, filtered_list,remaining_list)
    
    file_paths = [
        os.path.join('uncertain1/query_pool', 'x_subset.csv'),
        os.path.join(name,'x_subset_0.05.csv')
    ]
    merge_dataframes(file_paths, os.path.join(name,'x_subset.csv'))
    
if __name__ == "__main__":
    main()
