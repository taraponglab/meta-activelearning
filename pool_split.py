import os
import pandas as pd

# List of fingerprint files
fingerprint_files = [
    'AD2D.csv', 'AP2DC.csv', 'CDK.csv', 'CDKExt.csv',
    'CDKGraph.csv', 'EState.csv', 'KRFP.csv', 'KRFPC.csv',
    'MACCS.csv', 'PubChem.csv', 'SubFP.csv', 'SubFPC.csv'
]

# Paths
initial_pool_dir = 'data/initial_pool'
output_dir = 'data/new_pool'
ligand_list_csv = 'subset1/remaining_pool.csv'  # Update this to your LigandID list CSV of pool data you want to keep

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read LigandID list
ligand_df = pd.read_csv(ligand_list_csv)
ligand_ids = set(ligand_df['LigandID'].astype(str))

for fp_file in fingerprint_files:
    fp_path = os.path.join(initial_pool_dir, fp_file)
    if not os.path.exists(fp_path):
        print(f"File not found: {fp_path}")
        continue

    df = pd.read_csv(fp_path)
    # Filter rows where LigandID is in the list
    filtered_df = df[df['LigandID'].astype(str).isin(ligand_ids)]
    # Save to new folder
    out_path = os.path.join(output_dir, fp_file)
    filtered_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")