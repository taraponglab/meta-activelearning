import pandas as pd
import numpy as np
import os
import calculate_fp as cf
from tensorflow.keras.models import load_model

"""
Prediciton for pool data set with models in each sampling subsets.
"""

def split_pool(large_filepath, large_filename, list_filepath, list_filename, output_path, filtered_list,remaining_list):
    # Load the large data (assume the CSV has columns 'LigandID' and 'canonical_smiles')
    large_df = pd.read_csv(os.path.join(large_filepath, large_filename))
    compound_list_df = pd.read_csv(os.path.join(list_filepath, list_filename))

    # Check if 'LigandID' columns are in both DataFrames
    if 'LigandID' not in large_df.columns:
        print(f"'LigandID' column not found in {large_filename}")
    if 'LigandID' not in compound_list_df.columns:
        print(f"'LigandID' column not found in {list_filename}")

    large_df['LigandID'] = large_df['LigandID'].astype(str).str.strip()
    compound_list_df['LigandID'] = compound_list_df['LigandID'].astype(str).str.strip()
    filtered_list_df = large_df[large_df['LigandID'].isin(compound_list_df['LigandID'])]
    remaining_list_df = large_df[~large_df['LigandID'].isin(compound_list_df['LigandID'])]
    
    # Save the DataFrames to a new CSV file
    filtered_list_df.to_csv(os.path.join(output_path, filtered_list), index=False)
    remaining_list_df.to_csv(os.path.join(output_path, remaining_list), index=False)

    print("CSV files split generated successfully.")

def y_prediction(model, x_train_np, x_train, col_name):
    """
    Generates predictions and probabilities for a given model and input data.
    ------
    Parameters:
    model       : The trained model used for making predictions.
    x_train_np  : Input data in NumPy array format for the model.
    x_train     : Input data in DataFrame format, used to preserve indexing.
    col_name    : Column name for the output DataFrame representing predictions or probabilities.
    ------
    Returns:
    y_prob_df   : DataFrame containing the predicted probabilities.
    y_pred_df   : DataFrame containing the binary predictions.
    """
    y_prob = model.predict(x_train_np)
    y_pred = (y_prob > 0.5).astype(int).reshape(-1) 
    y_prob = y_prob.reshape(-1, y_prob.shape[-1])
    y_prob_df = pd.DataFrame(y_prob, columns=[col_name]).set_index(x_train.index)
    y_pred_df = pd.DataFrame(y_pred, columns=[col_name]).set_index(x_train.index)
    
    return y_prob_df, y_pred_df

def load(name, pool_file, model_file):
    """
    Running prediction with provided stack ensemble learning model.
    ------
    Parameters:
    name    : The directory where the input file is located and where the output files will be saved.
    """
    xat_data = pd.read_csv(os.path.join( pool_file,  'AD2D.csv'     ), index_col=0)
    xes_data = pd.read_csv(os.path.join( pool_file,  'EState.csv'   ), index_col=0)
    xke_data = pd.read_csv(os.path.join( pool_file,  'KRFP.csv'     ), index_col=0)
    xpc_data = pd.read_csv(os.path.join( pool_file,  'PubChem.csv'  ), index_col=0)
    xss_data = pd.read_csv(os.path.join( pool_file,  'SubFP.csv'    ), index_col=0)
    xcd_data = pd.read_csv(os.path.join( pool_file,  'CDKGraph.csv' ), index_col=0)
    xcn_data = pd.read_csv(os.path.join( pool_file,  'CDK.csv'      ), index_col=0)
    xkc_data = pd.read_csv(os.path.join( pool_file,  'KRFPC.csv'    ), index_col=0)
    xce_data = pd.read_csv(os.path.join( pool_file,  'CDKExt.csv'   ), index_col=0)
    xsc_data = pd.read_csv(os.path.join( pool_file,  'SubFPC.csv'   ), index_col=0)
    xac_data = pd.read_csv(os.path.join( pool_file,  'AP2DC.csv'    ), index_col=0)
    xma_data = pd.read_csv(os.path.join( pool_file,  'MACCS.csv'    ), index_col=0)

    xat_data_np = np.array(xat_data)
    xes_data_np = np.array(xes_data)
    xke_data_np = np.array(xke_data)
    xpc_data_np = np.array(xpc_data)
    xss_data_np = np.array(xss_data)
    xcd_data_np = np.array(xcd_data)
    xcn_data_np = np.array(xcn_data)
    xkc_data_np = np.array(xkc_data)
    xce_data_np = np.array(xce_data)
    xsc_data_np = np.array(xsc_data)
    xac_data_np = np.array(xac_data)
    xma_data_np = np.array(xma_data)
    
    # CNN architecture
    baseline_model_cnn_at = load_model(os.path.join(model_file, "baseline_model_cnn_at.keras"))
    baseline_model_cnn_es = load_model(os.path.join(model_file, "baseline_model_cnn_es.keras"))
    baseline_model_cnn_ke = load_model(os.path.join(model_file, "baseline_model_cnn_ke.keras"))
    baseline_model_cnn_pc = load_model(os.path.join(model_file, "baseline_model_cnn_pc.keras"))
    baseline_model_cnn_ss = load_model(os.path.join(model_file, "baseline_model_cnn_ss.keras"))
    baseline_model_cnn_cd = load_model(os.path.join(model_file, "baseline_model_cnn_cd.keras"))
    baseline_model_cnn_cn = load_model(os.path.join(model_file, "baseline_model_cnn_cn.keras"))
    baseline_model_cnn_kc = load_model(os.path.join(model_file, "baseline_model_cnn_kc.keras"))
    baseline_model_cnn_ce = load_model(os.path.join(model_file, "baseline_model_cnn_ce.keras"))
    baseline_model_cnn_sc = load_model(os.path.join(model_file, "baseline_model_cnn_sc.keras"))
    baseline_model_cnn_ac = load_model(os.path.join(model_file, "baseline_model_cnn_ac.keras"))
    baseline_model_cnn_ma = load_model(os.path.join(model_file, "baseline_model_cnn_ma.keras"))

    yat_prob_cnn, yat_pred_cnn   =  y_prediction(baseline_model_cnn_at, xat_data_np, xat_data, "yat_pred_cnn")
    yes_prob_cnn, yes_pred_cnn   =  y_prediction(baseline_model_cnn_es, xes_data_np, xes_data, "yes_pred_cnn")
    yke_prob_cnn, yke_pred_cnn   =  y_prediction(baseline_model_cnn_ke, xke_data_np, xke_data, "yke_pred_cnn")
    ypc_prob_cnn, ypc_pred_cnn   =  y_prediction(baseline_model_cnn_pc, xpc_data_np, xpc_data, "ypc_pred_cnn")
    yss_prob_cnn, yss_pred_cnn   =  y_prediction(baseline_model_cnn_ss, xss_data_np, xss_data, "yss_pred_cnn")
    ycd_prob_cnn, ycd_pred_cnn   =  y_prediction(baseline_model_cnn_cd, xcd_data_np, xcd_data, "ycd_pred_cnn")
    ycn_prob_cnn, ycn_pred_cnn   =  y_prediction(baseline_model_cnn_cn, xcn_data_np, xcn_data, "ycn_pred_cnn")
    ykc_prob_cnn, ykc_pred_cnn   =  y_prediction(baseline_model_cnn_kc, xkc_data_np, xkc_data, "ykc_pred_cnn")
    yce_prob_cnn, yce_pred_cnn   =  y_prediction(baseline_model_cnn_ce, xce_data_np, xce_data, "yce_pred_cnn")
    ysc_prob_cnn, ysc_pred_cnn   =  y_prediction(baseline_model_cnn_sc, xsc_data_np, xsc_data, "ysc_pred_cnn")
    yac_prob_cnn, yac_pred_cnn   =  y_prediction(baseline_model_cnn_ac, xac_data_np, xac_data, "yac_pred_cnn")
    yma_prob_cnn, yma_pred_cnn   =  y_prediction(baseline_model_cnn_ma, xma_data_np, xma_data, "yma_pred_cnn")
    
    # BiLSTM architecture
    xat_data_np_bilstm = xat_data_np.reshape((-1, 1, xat_data_np.shape[1]))
    xes_data_np_bilstm = xes_data_np.reshape((-1, 1, xes_data_np.shape[1]))
    xke_data_np_bilstm = xke_data_np.reshape((-1, 1, xke_data_np.shape[1]))
    xpc_data_np_bilstm = xpc_data_np.reshape((-1, 1, xpc_data_np.shape[1]))
    xss_data_np_bilstm = xss_data_np.reshape((-1, 1, xss_data_np.shape[1]))
    xcd_data_np_bilstm = xcd_data_np.reshape((-1, 1, xcd_data_np.shape[1]))
    xcn_data_np_bilstm = xcn_data_np.reshape((-1, 1, xcn_data_np.shape[1]))
    xkc_data_np_bilstm = xkc_data_np.reshape((-1, 1, xkc_data_np.shape[1]))
    xce_data_np_bilstm = xce_data_np.reshape((-1, 1, xce_data_np.shape[1]))
    xsc_data_np_bilstm = xsc_data_np.reshape((-1, 1, xsc_data_np.shape[1]))
    xac_data_np_bilstm = xac_data_np.reshape((-1, 1, xac_data_np.shape[1]))
    xma_data_np_bilstm = xma_data_np.reshape((-1, 1, xma_data_np.shape[1]))
    
    baseline_model_bilstm_at = load_model(os.path.join(model_file, "baseline_model_bilstm_at.keras"))
    baseline_model_bilstm_es = load_model(os.path.join(model_file, "baseline_model_bilstm_es.keras"))
    baseline_model_bilstm_ke = load_model(os.path.join(model_file, "baseline_model_bilstm_ke.keras"))
    baseline_model_bilstm_pc = load_model(os.path.join(model_file, "baseline_model_bilstm_pc.keras"))
    baseline_model_bilstm_ss = load_model(os.path.join(model_file, "baseline_model_bilstm_ss.keras"))
    baseline_model_bilstm_cd = load_model(os.path.join(model_file, "baseline_model_bilstm_cd.keras"))
    baseline_model_bilstm_cn = load_model(os.path.join(model_file, "baseline_model_bilstm_cn.keras"))
    baseline_model_bilstm_kc = load_model(os.path.join(model_file, "baseline_model_bilstm_kc.keras"))
    baseline_model_bilstm_ce = load_model(os.path.join(model_file, "baseline_model_bilstm_ce.keras"))
    baseline_model_bilstm_sc = load_model(os.path.join(model_file, "baseline_model_bilstm_sc.keras"))
    baseline_model_bilstm_ac = load_model(os.path.join(model_file, "baseline_model_bilstm_ac.keras"))
    baseline_model_bilstm_ma = load_model(os.path.join(model_file, "baseline_model_bilstm_ma.keras"))
    
    yat_prob_bilstm, yat_pred_bilstm   = y_prediction(  baseline_model_bilstm_at, xat_data_np_bilstm, xat_data,   "yat_pred_bilstm")
    yes_prob_bilstm, yes_pred_bilstm   = y_prediction(  baseline_model_bilstm_es, xes_data_np_bilstm, xes_data,   "yes_pred_bilstm")
    yke_prob_bilstm, yke_pred_bilstm   = y_prediction(  baseline_model_bilstm_ke, xke_data_np_bilstm, xke_data,   "yke_pred_bilstm")
    ypc_prob_bilstm, ypc_pred_bilstm   = y_prediction(  baseline_model_bilstm_pc, xpc_data_np_bilstm, xpc_data,   "ypc_pred_bilstm")
    yss_prob_bilstm, yss_pred_bilstm   = y_prediction(  baseline_model_bilstm_ss, xss_data_np_bilstm, xss_data,   "yss_pred_bilstm")
    ycd_prob_bilstm, ycd_pred_bilstm   = y_prediction(  baseline_model_bilstm_cd, xcd_data_np_bilstm, xcd_data,   "ycd_pred_bilstm")
    ycn_prob_bilstm, ycn_pred_bilstm   = y_prediction(  baseline_model_bilstm_cn, xcn_data_np_bilstm, xcn_data,   "ycn_pred_bilstm")
    ykc_prob_bilstm, ykc_pred_bilstm   = y_prediction(  baseline_model_bilstm_kc, xkc_data_np_bilstm, xkc_data,   "ykc_pred_bilstm")
    yce_prob_bilstm, yce_pred_bilstm   = y_prediction(  baseline_model_bilstm_ce, xce_data_np_bilstm, xce_data,   "yce_pred_bilstm")
    ysc_prob_bilstm, ysc_pred_bilstm   = y_prediction(  baseline_model_bilstm_sc, xsc_data_np_bilstm, xsc_data,   "ysc_pred_bilstm")
    yac_prob_bilstm, yac_pred_bilstm   = y_prediction(  baseline_model_bilstm_ac, xac_data_np_bilstm, xac_data,   "yac_pred_bilstm")
    yma_prob_bilstm, yma_pred_bilstm   = y_prediction(  baseline_model_bilstm_ma, xma_data_np_bilstm, xma_data,   "yma_pred_bilstm")
    
    # Attention architecture
    baseline_model_att_at = load_model(os.path.join(model_file, "baseline_model_att_at.keras"))
    baseline_model_att_es = load_model(os.path.join(model_file, "baseline_model_att_es.keras"))
    baseline_model_att_ke = load_model(os.path.join(model_file, "baseline_model_att_ke.keras"))
    baseline_model_att_pc = load_model(os.path.join(model_file, "baseline_model_att_pc.keras"))
    baseline_model_att_ss = load_model(os.path.join(model_file, "baseline_model_att_ss.keras"))
    baseline_model_att_cd = load_model(os.path.join(model_file, "baseline_model_att_cd.keras"))
    baseline_model_att_cn = load_model(os.path.join(model_file, "baseline_model_att_cn.keras"))
    baseline_model_att_kc = load_model(os.path.join(model_file, "baseline_model_att_kc.keras"))
    baseline_model_att_ce = load_model(os.path.join(model_file, "baseline_model_att_ce.keras"))
    baseline_model_att_sc = load_model(os.path.join(model_file, "baseline_model_att_sc.keras"))
    baseline_model_att_ac = load_model(os.path.join(model_file, "baseline_model_att_ac.keras"))
    baseline_model_att_ma = load_model(os.path.join(model_file, "baseline_model_att_ma.keras"))
    
    yat_prob_att, yat_pred_att   = y_prediction(  baseline_model_att_at, xat_data_np, xat_data,   "yat_pred_att")
    yes_prob_att, yes_pred_att   = y_prediction(  baseline_model_att_es, xes_data_np, xes_data,   "yes_pred_att")
    yke_prob_att, yke_pred_att   = y_prediction(  baseline_model_att_ke, xke_data_np, xke_data,   "yke_pred_att")
    ypc_prob_att, ypc_pred_att   = y_prediction(  baseline_model_att_pc, xpc_data_np, xpc_data,   "ypc_pred_att")
    yss_prob_att, yss_pred_att   = y_prediction(  baseline_model_att_ss, xss_data_np, xss_data,   "yss_pred_att")
    ycd_prob_att, ycd_pred_att   = y_prediction(  baseline_model_att_cd, xcd_data_np, xcd_data,   "ycd_pred_att")
    ycn_prob_att, ycn_pred_att   = y_prediction(  baseline_model_att_cn, xcn_data_np, xcn_data,   "ycn_pred_att")
    ykc_prob_att, ykc_pred_att   = y_prediction(  baseline_model_att_kc, xkc_data_np, xkc_data,   "ykc_pred_att")
    yce_prob_att, yce_pred_att   = y_prediction(  baseline_model_att_ce, xce_data_np, xce_data,   "yce_pred_att")
    ysc_prob_att, ysc_pred_att   = y_prediction(  baseline_model_att_sc, xsc_data_np, xsc_data,   "ysc_pred_att")
    yac_prob_att, yac_pred_att   = y_prediction(  baseline_model_att_ac, xac_data_np, xac_data,   "yac_pred_att")
    yma_prob_att, yma_pred_att   = y_prediction(  baseline_model_att_ma, xma_data_np, xma_data,   "yma_pred_att")
    
    # Save predictive features
    stack_data_prob_all = pd.concat([yat_prob_cnn, yat_prob_bilstm, yat_prob_att,
                            yes_prob_cnn, yes_prob_bilstm, yes_prob_att,
                            yke_prob_cnn, yke_prob_bilstm, yke_prob_att,
                            ypc_prob_cnn, ypc_prob_bilstm, ypc_prob_att,
                            yss_prob_cnn, yss_prob_bilstm, yss_prob_att,
                            ycd_prob_cnn, ycd_prob_bilstm, ycd_prob_att,
                            ycn_prob_cnn, ycn_prob_bilstm, ycn_prob_att,
                            ykc_prob_cnn, ykc_prob_bilstm, ykc_prob_att,
                            yce_prob_cnn, yce_prob_bilstm, yce_prob_att,
                            ysc_prob_cnn, ysc_prob_bilstm, ysc_prob_att,
                            yac_prob_cnn, yac_prob_bilstm, yac_prob_att,
                            yma_prob_cnn, yma_prob_bilstm, yma_prob_att],  axis=1)
    stack_data_pred_all  = pd.concat([yat_pred_cnn, yat_pred_bilstm, yat_pred_att,
                        yes_pred_cnn, yes_pred_bilstm, yes_pred_att,
                        yke_pred_cnn, yke_pred_bilstm, yke_pred_att,
                        ypc_pred_cnn, ypc_pred_bilstm, ypc_pred_att,
                        yss_pred_cnn, yss_pred_bilstm, yss_pred_att,
                        ycd_pred_cnn, ycd_pred_bilstm, ycd_pred_att,
                        ycn_pred_cnn, ycn_pred_bilstm, ycn_pred_att,
                        ykc_pred_cnn, ykc_pred_bilstm, ykc_pred_att,
                        yce_pred_cnn, yce_pred_bilstm, yce_pred_att,
                        ysc_pred_cnn, ysc_pred_bilstm, ysc_pred_att,
                        yac_pred_cnn, yac_pred_bilstm, yac_pred_att,
                        yma_pred_cnn, yma_pred_bilstm, yma_pred_att],  axis=1)
    
    stack_data_prob_all.to_csv(os.path.join(name, "all_stacked_pool_prob.csv"))
    stack_data_pred_all.to_csv(os.path.join(name, "all_stacked_pool_predict.csv"))

def main():
    # Generate reamining pool data set with different fingerprints
    large_filenames = [
    'AD2D.csv', 'AP2DC.csv', 'CDK.csv', 'CDKExt.csv', 
    'CDKGraph.csv', 'EState.csv', 'KRFP.csv', 'KRFPC.csv', 
    'MACCS.csv', 'PubChem.csv', 'SubFP.csv', 'SubFPC.csv'
    ]

    large_filepath = 'margin/pool_pred/pool_1337'
    list_filepath = 'margin/pool_pred/pool_1271'
    list_filename = 'remaining_pool.csv'
    output_path = 'margin/pool_pred/pool_1271'

    for large_filename in large_filenames:
        # Use the same filename for filtered_list
        filtered_list = large_filename  # Assuming you want the same filename as filtered_list

        # Call your function
        split_pool(
            large_filepath,
            large_filename,
            list_filepath,
            list_filename,
            output_path,
            filtered_list
        )
    print(f"Processed {large_filename}")

    for i in range(1, 7):  # Iterating from train_1 to train_6
        folder_name = f'margin1/train_{i}'  # Generate folder name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")
        print("#" * 100)
        print(folder_name)
        load(folder_name, pool_file='margin/pool_pred/pool_1337', model_file=folder_name)
        print("Finish predict!")

if __name__ == "__main__":
    main()