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

def split_data(large_filename, list_filepath, list_filename, output_path, filtered_list):
    large_df = pd.read_csv(large_filename)       # Load the large data
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
    name = input("Enter the name of the directory (e.g., 'subset1'): ").strip()
    path_file = 'data/initial_pool' #input("Enter the path to the pool prediction file (e.g., 'data/initial_pool'): ").strip()
    training_file = 'data/x_train.csv'  #input("Enter the path to the training file (e.g., 'data/x_train.csv'): ").strip()
    prev_subset = 'data/subsets/x_subset_1.csv' #input("Enter the path to the previous subset file (e.g., 'data/subsets/x_subset_1.csv'): ").strip()

    # Step 1: Read probability file and calculate binary probabilities
    prob_path = os.path.join(name, 'meta_pool_cnn', 'y_prob_pool_average_cnn.csv')
    if not os.path.exists(prob_path):
        print(f"File not found: {prob_path}")
        return
    df = pd.read_csv(prob_path, index_col=0)
    get_prob_0(df, name)

    # Step 2: Calculate uncertainty
    binary_prob_path = os.path.join(name, "y_prob_pool_average_cnn_binary.csv")
    if not os.path.exists(binary_prob_path):
        print(f"File not found: {binary_prob_path}")
        return
    df = pd.read_csv(binary_prob_path, index_col=0)
    print(df.dtypes)
    uncertainty(name, df)

    # Step 3: Sort by uncertainty and split
    uncertain_path = os.path.join(name, "uncertain_prob.csv")
    if not os.path.exists(uncertain_path):
        print(f"File not found: {uncertain_path}")
        return
    uncertain_cal_file = pd.read_csv(uncertain_path, index_col=0)
    uncertain_sort(name, uncertain_cal_file)

    # Step 4: Split y_pool based on uncertain subset
    y_pool_path = os.path.join(path_file, "y_pool.csv")
    if not os.path.exists(y_pool_path):
        print(f"File not found: {y_pool_path}")
        return
    y_pool = pd.read_csv(y_pool_path)
    uncertain_subset_path = os.path.join(name, "uncertain_subset.csv")
    if not os.path.exists(uncertain_subset_path):
        print(f"File not found: {uncertain_subset_path}")
        return
    uncertain_subset_df = pd.read_csv(uncertain_subset_path)
    split_y_pool(name, y_pool, uncertain_subset_df)

    # Step 5: Split x_train based on uncertain subset
    if not os.path.exists(training_file):
        print(f"File not found: {training_file}")
        return
    list_filepath = name
    list_filename = 'uncertain_subset.csv'
    output_path = name
    filtered_list = 'x_subset_0.05.csv'
    split_data(training_file, list_filepath, list_filename, output_path, filtered_list)

    # Step 6: Merge with previous query pool if exists
    new_subset = os.path.join(name, 'x_subset_0.05.csv')
    merged_output = os.path.join(name, 'x_subset.csv')
    file_paths = []
    if os.path.exists(prev_subset):
        file_paths.append(prev_subset)
    if os.path.exists(new_subset):
        file_paths.append(new_subset)
    if file_paths:
        merge_dataframes(file_paths, merged_output)
        print(f"Merged x_subset.csv saved to {merged_output}")
    else:
        print("No files found to merge for x_subset.csv.")
    
if __name__ == "__main__":
    main()
