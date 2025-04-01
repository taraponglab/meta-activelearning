import pandas as pd
import numpy as np
import os

def get_prob_0(name, df):
    if 'y_prob_pool_average' in df.columns:
        # Class 0 probabilities
        df['y_prob_pool_0'] = 1 - df['y_prob_pool_average']
        # Save the updated DataFrame to a new CSV file
        df.to_csv(os.path.join(name, 'y_prob_pool_average_cnn_binary.csv'), index=True)
        print("The updated CSV file 'y_prob_pool_average_cnn_binary.csv' has been created successfully.")
    else:
        print("The column 'y_prob_pool_average' does not exist in the dataset.")

def margin(name, df):
    proba = df.values   # Convert the DataFrame to a NumPy array for easier manipulation
    # Calculate the margin for each row
    part = np.partition(-proba, 1, axis=1)
    margins = -part[:, 0] + part[:, 1]
    df['y_prob_margin'] = margins  # Create a new DataFrame for the margins
    df.to_csv(os.path.join(name, 'margin_prob.csv' ), index=True)
    print(f"Margins have been calculated and saved.")

def margin_sort(name, df, percentage):
    df_sorted = df.sort_values(by='y_prob_margin', ascending=True) # Sort compounds based on the margin
    n = int(len(df_sorted) * percentage)    # Determine the number of top percentage rows to select
    # Split the data into two sets
    margin_subset  = df_sorted.iloc[:n]
    remaining_data = df_sorted.iloc[n:]
    # Save the datasets
    margin_subset.to_csv(os.path.join(name, "margin_subset.csv"), index=True)
    remaining_data.to_csv(os.path.join(name, "remaining_pool.csv"), index=True)
    
    print(f"Split completed: {n} rows saved in 'margin_subset.csv' and {len(df_sorted) - n} in 'remaining_pool.csv'.")

def split_y_pool(name, df, margin_subset_df):
    margin_y_pool = df[df["LigandID"].isin(margin_subset_df["LigandID"])]
    remaining_y_pool = df[~df["LigandID"].isin(margin_subset_df["LigandID"])]
    margin_y_pool.to_csv(os.path.join(name, "margin_subset_y_pool.csv"), index=False)
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
    """
    Merge multiple CSV files into a single DataFrame and save it.
    
    Parameters:
    file_paths (list of str): List of file paths to CSV files to be merged.
    output_path (str): Path to save the merged DataFrame.
    how (str): Type of merge to be performed. Options are 'inner', 'outer', 'left', or 'right'. Default is 'outer'.
    """
    # Read and merge the DataFrames
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    df_merged = pd.concat(dfs, axis=0, ignore_index=True, join=how)
    
    df_merged.to_csv(output_path, index=False)      # Save the merged DataFrame to CSV

def main():
    name = "margin1"
    path_file = 'pool_pred/pool_1337'
    df = pd.read_csv(os.path.join(name, 'meta_pool_cnn', 'y_prob_pool_average_cnn.csv'), index_col=0)
    get_prob_0(name, df)
    df = pd.read_csv(os.path.join(name, "y_prob_pool_average_cnn_binary.csv"), index_col=0)
    margin(name, df)
    margin_cal_file = pd.read_csv(os.path.join(name, "margin_prob.csv"), index_col=0)
    margin_sort(name, margin_cal_file, percentage=0.05)
    y_pool = pd.read_csv(os.path.join(path_file, "y_pool.csv"))
    margin_subset_df = pd.read_csv(os.path.join(name, "margin_subset.csv"))
    split_y_pool(name, y_pool, margin_subset_df)

    large_filepath = 'data'
    large_filename = 'tpo_combine_process.csv'
    list_filepath = name
    list_filename = 'margin_subset.csv'
    output_path = name
    filtered_list = 'x_subset_0.05.csv'
    split_data(large_filepath, large_filename, list_filepath, list_filename, output_path, filtered_list)
    
    file_paths = [
        os.path.join('margin1/query_pool', 'x_subset.csv'),
        os.path.join(name,'x_subset_0.05.csv')
    ]
    merge_dataframes(file_paths, os.path.join(name,'x_subset.csv'))
    
if __name__ == "__main__":
    main()
