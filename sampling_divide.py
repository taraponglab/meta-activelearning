import os
import pandas as pd
import calculate_fp as cf

def separate_classes(df, name):
    '''
    Separates the DataFrame into two classes based on the 'TC TPO inhibition activity' column.
    ------
    Parameters:
    df      : DataFrame containing the chemical data.
    name    : The directory where the input file is located and where the output files will be saved.
    ------
    Returns:
    active_df   : DataFrame containing active compounds.
    inactive_df : DataFrame containing inactive compounds.
    '''
    activity_column = "TC TPO inhibition activity"
    active_df = df[df[activity_column] == "Active"]
    inactive_df = df[df[activity_column] == "Inactive"]

    # Save to separate CSV files
    active_df.to_csv(os.path.join(name, "active_compounds.csv"), index=False)
    inactive_df.to_csv(os.path.join(name, "inactive_compounds.csv"), index=False)

    print(f"Active compounds: {len(active_df)}")
    print(f"Inactive compounds: {len(inactive_df)}")
    return active_df, inactive_df

def divide_ratio(df, name):
    '''
    Divides the inactive compounds DataFrame into 6 equal parts and saves them as separate CSV files.
    ------
    Parameters:
    df   : DataFrame containing inactive compounds.
    name : The directory where the input file is located and where the output files will be saved.
    '''
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  #Shuffle the whole dataset (frac=1 ensures no rows are lost)
    # Determine split sizes
    num_splits = 6
    split_sizes = [len(df) // num_splits] * num_splits  # Start with equal sizes
    for i in range(len(df) % num_splits):  # Distribute the remainder
        split_sizes[i] += 1  

    # Split dataset accordingly
    datasets = []
    start_idx = 0
    for size in split_sizes:
        datasets.append(df.iloc[start_idx:start_idx + size])
        start_idx += size

    # Save each part as a separate CSV file
    for i, data in enumerate(datasets):
        data.to_csv(os.path.join(name, f"inactive_set_{i+1}.csv"), index=False)

    print([len(data) for data in datasets]) 

def combine_each_set(active_df, name):
    '''
    Combines the active compounds with each of the inactive dataset splits and saves them as new training sets.
    ------
    Parameters:
    active_df : DataFrame containing active compounds.
    name      : The directory where the input file is located and where the output files will be saved.
    ------
    Returns:
    combined_df : DataFrame containing the combined active and inactive compounds for the first split.
    '''
    # Load the inactive dataset splits
    num_splits = 6
    inactive_datasets = [pd.read_csv(os.path.join(name, f"inactive_set_{i+1}.csv")) for i in range(num_splits)]

    # Create different combinations (each inactive split + active compounds)
    for i in range(num_splits):
        combined_df = pd.concat([active_df, inactive_datasets[i]], ignore_index=True)  # Combine active with one split
        combined_df.to_csv(os.path.join(name, f"train_set_{i+1}.csv"), index=False)  # Save the combined file
        print(f"train_set_{i+1}.csv: {len(combined_df)} compounds")
    
    return combined_df

def calculate_fingerprint(df, name):
    """
    Calculates fingerprints for chemical compounds based on a given DataFrame.
    ------
    Parameters:
    df      : The filename of the CSV file containing the chemical data.
    name    : The directory where the input file is located and where the output files will be saved.
    ------
    Returns:
    None: Outputs are saved directly to files in the specified directory.
    """
    output_smi_file=cf.convert_to_smi(df, name, column='canonical_smiles')
    FP_list, fp=cf.read_descriptor(folder_path = "fingerprints_xml")
    cf.calculate_fp(FP_list, df, output_smi_file, fp, fingerprint_output_dir=name)

def extract_all_fingerprints(subset_df, big_fp_folder, output_folder, fp_list=None):
    """
    Extracts all 12 fingerprints for the given subset from the big fingerprint CSV files.
    subset_df: DataFrame with at least 'LigandID' column.
    big_fp_folder: Folder containing the big fingerprint CSV files (one per fingerprint).
    output_folder: Folder to save the extracted fingerprint CSVs.
    fp_list: List of fingerprint names (default: the 12 you provided).
    """
    if fp_list is None:
        fp_list = [
            'AP2DC', 'AD2D', 'EState', 'CDKExt', 'CDK', 'CDKGraph',
            'KRFPC', 'KRFP', 'MACCS', 'PubChem', 'SubFPC', 'SubFP'
        ]
    os.makedirs(output_folder, exist_ok=True)
    for fp in fp_list:
        big_fp_path = os.path.join(big_fp_folder, f"{fp}.csv")
        output_fp_path = os.path.join(output_folder, f"{fp}.csv")
        big_fp_df = pd.read_csv(big_fp_path)
        merged = pd.merge(subset_df[['LigandID']], big_fp_df, on='LigandID', how='left')
        merged.to_csv(output_fp_path, index=False)
        print(f"Extracted {fp} fingerprints saved to {output_fp_path}")

def genarate_y(df, name):
    '''
    Generates a target variable DataFrame from the input DataFrame and saves it as 'y_train.csv'.
    '''
    df = df[['LigandID', 'TC TPO inhibition activity']]
    df['TC TPO inhibition activity'] = df['TC TPO inhibition activity'].replace({'Active': 1, 'Inactive': 0})
    df = df.rename(columns={'TC TPO inhibition activity': 'Class'})
    df.to_csv(os.path.join(name, 'y_train.csv'), index=False)
    print("Conversion complete. The converted file is saved as 'y_train.csv'.")

def main():
    file_name = input("Type your CSV file name (including extension): ")
    name = input("Type your folder path name: ")
    # Ensure the output directory exists
    if not os.path.exists(name):
        os.makedirs(name)
        print(f"Folder '{name}' created.")
    # Read the CSV file
    df = pd.read_csv(os.path.join(file_name))
    active_df, inactive_df=separate_classes(df, name)
    divide_ratio(inactive_df, name)
    combine_each_set(active_df, name)
    for i in range(6):
        folder_name = os.path.join(name, f"train_{i+1}")  # Use a dynamic name for each dataset
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")
        combined_df = pd.read_csv(os.path.join(name, f"train_set_{i+1}.csv"))
        # Use when necessary
        # calculate_fingerprint(combined_df, folder_name)
        # Extract fingerprints from the big fingerprint CSV files
        extract_all_fingerprints(combined_df, "data/train", folder_name)
        genarate_y(combined_df, folder_name)

if __name__ == "__main__":
    main() 