import tensorflow.keras as keras
import pandas as pd
import numpy as np
import os
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, precision_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

"""
Average probability prediciton for pool data set.
"""

def y_prediction(model, x_train_np, x_train, y_train, col_name):
    y_prob = model.predict(x_train_np)
    y_pred = (y_prob > 0.5).astype(int).reshape(-1)
    print(y_train.shape)
    print(y_pred.shape)
    print(y_pred)
    acc = accuracy_score(y_train, y_pred)
    sen = recall_score(y_train, y_pred)  # Sensitivity is the same as recall
    mcc = matthews_corrcoef(y_train, y_pred)
    f1  = f1_score(y_train, y_pred)
    y_prob = y_prob.reshape(-1, y_prob.shape[-1])
    print(y_prob.shape)
    
    auc = roc_auc_score(y_train, y_prob)
    bcc = balanced_accuracy_score(y_train, y_pred)
    pre = precision_score(y_train, y_pred)
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spc = tn / (tn + fp)
    av_pre = average_precision_score(y_train, y_prob)
    
    y_prob = pd.DataFrame(y_prob, columns=[col_name]).set_index(x_train.index)
    y_pred_df = pd.DataFrame(y_pred, columns=[col_name]).set_index(x_train.index)

    # Create a DataFrame to store the metrics
    metrics = pd.DataFrame({
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'F1 Score': [f1],
        'AUC': [auc],
        'BACC': [bcc],
        'Precision': [pre],
        'Average Precision': [av_pre]
    }, index=[col_name])
    
    return y_prob, y_pred_df, metrics

def y_prediction_average(y_prob, y_train, col_name):
    y_pred = (y_prob > 0.5).astype(int)
    print(y_train.shape)
    print(y_prob.shape)
    print(y_pred.shape)
    print(y_pred)
    acc = accuracy_score(y_train, y_pred)
    sen = recall_score(y_train, y_pred)  # Sensitivity is the same as recall
    mcc = matthews_corrcoef(y_train, y_pred)
    f1  = f1_score(y_train, y_pred)
    
    auc = roc_auc_score(y_train, y_prob)
    bcc = balanced_accuracy_score(y_train, y_pred)
    pre = precision_score(y_train, y_pred)
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spc = tn / (tn + fp)
    av_pre = average_precision_score(y_train, y_prob)
    
    y_pred_df = pd.DataFrame(y_pred)

    # Create a DataFrame to store the metrics
    metrics = pd.DataFrame({
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'F1 Score': [f1],
        'AUC': [auc],
        'BACC': [bcc],
        'Precision': [pre],
        'Average Precision': [av_pre]
    }, index=[col_name])
    
    return y_pred_df, metrics

def meta_cnn_train(name, model_file):
    '''
    This function is to train the meta-model based CNN
    '''
    
    x1_pool = pd.read_csv(os.path.join( 'margin1', 'train_1',  "all_stacked_pool_prob.csv"  ), index_col=0)
    x2_pool = pd.read_csv(os.path.join( 'margin1', 'train_2',  "all_stacked_pool_prob.csv"  ), index_col=0)
    x3_pool = pd.read_csv(os.path.join( 'margin1', 'train_3',  "all_stacked_pool_prob.csv"  ), index_col=0)
    x4_pool = pd.read_csv(os.path.join( 'margin1', 'train_4',  "all_stacked_pool_prob.csv"  ), index_col=0)
    x5_pool = pd.read_csv(os.path.join( 'margin1', 'train_5',  "all_stacked_pool_prob.csv"  ), index_col=0)
    x6_pool = pd.read_csv(os.path.join( 'margin1', 'train_6',  "all_stacked_pool_prob.csv"  ), index_col=0)
    y_pool = pd.read_csv(os.path.join( 'pool_pred', 'pool_1337', "y_pool.csv"  ), index_col=0)
    
    x1_pool_np = np.array(x1_pool)
    x2_pool_np = np.array(x2_pool)
    x3_pool_np = np.array(x3_pool)
    x4_pool_np = np.array(x4_pool)
    x5_pool_np = np.array(x5_pool)
    x6_pool_np = np.array(x6_pool)
    y_pool = np.array(y_pool)
    
    meta_model_cnn_1 = load_model(os.path.join(model_file, "meta_model_cnn_1.keras"))
    meta_model_cnn_2 = load_model(os.path.join(model_file, "meta_model_cnn_2.keras"))
    meta_model_cnn_3 = load_model(os.path.join(model_file, "meta_model_cnn_3.keras"))
    meta_model_cnn_4 = load_model(os.path.join(model_file, "meta_model_cnn_4.keras"))
    meta_model_cnn_5 = load_model(os.path.join(model_file, "meta_model_cnn_5.keras"))
    meta_model_cnn_6 = load_model(os.path.join(model_file, "meta_model_cnn_6.keras"))
    
    y1_prob_cnn_pool, y1_pred_cnn_pool, y1_metric_cnn_pool   =  y_prediction(meta_model_cnn_1, x1_pool_np, x1_pool, y_pool, "y1_pred_cnn")
    y2_prob_cnn_pool, y2_pred_cnn_pool, y2_metric_cnn_pool   =  y_prediction(meta_model_cnn_2, x2_pool_np, x2_pool, y_pool, "y2_pred_cnn")
    y3_prob_cnn_pool, y3_pred_cnn_pool, y3_metric_cnn_pool   =  y_prediction(meta_model_cnn_3, x3_pool_np, x3_pool, y_pool, "y3_pred_cnn")
    y4_prob_cnn_pool, y4_pred_cnn_pool, y4_metric_cnn_pool   =  y_prediction(meta_model_cnn_4, x4_pool_np, x4_pool, y_pool, "y4_pred_cnn")
    y5_prob_cnn_pool, y5_pred_cnn_pool, y5_metric_cnn_pool   =  y_prediction(meta_model_cnn_5, x5_pool_np, x5_pool, y_pool, "y5_pred_cnn")
    y6_prob_cnn_pool, y6_pred_cnn_pool, y6_metric_cnn_pool   =  y_prediction(meta_model_cnn_6, x6_pool_np, x6_pool, y_pool, "y6_pred_cnn")
    
    y1_prob_cnn_pool.to_csv(os.path.join( name, "y1_prob_pool_cnn.csv"))
    y2_prob_cnn_pool.to_csv(os.path.join( name, "y2_prob_pool_cnn.csv"))
    y3_prob_cnn_pool.to_csv(os.path.join( name, "y3_prob_pool_cnn.csv"))
    y4_prob_cnn_pool.to_csv(os.path.join( name, "y4_prob_pool_cnn.csv"))
    y5_prob_cnn_pool.to_csv(os.path.join( name, "y5_prob_pool_cnn.csv"))
    y6_prob_cnn_pool.to_csv(os.path.join( name, "y6_prob_pool_cnn.csv"))
    y1_pred_cnn_pool.to_csv(os.path.join( name, "y1_pred_pool_cnn.csv"))
    y2_pred_cnn_pool.to_csv(os.path.join( name, "y2_pred_pool_cnn.csv"))
    y3_pred_cnn_pool.to_csv(os.path.join( name, "y3_pred_pool_cnn.csv"))
    y4_pred_cnn_pool.to_csv(os.path.join( name, "y4_pred_pool_cnn.csv"))
    y5_pred_cnn_pool.to_csv(os.path.join( name, "y5_pred_pool_cnn.csv"))
    y6_pred_cnn_pool.to_csv(os.path.join( name, "y6_pred_pool_cnn.csv"))
    
    stack_pool_prob = pd.concat([y1_prob_cnn_pool,
                        y2_prob_cnn_pool,
                        y3_prob_cnn_pool,
                        y4_prob_cnn_pool,
                        y5_prob_cnn_pool,
                        y6_prob_cnn_pool],  axis=1)
    stack_pool_pred = pd.concat([y1_pred_cnn_pool,
                    y2_pred_cnn_pool,
                    y3_pred_cnn_pool,
                    y4_pred_cnn_pool,
                    y5_pred_cnn_pool,
                    y6_pred_cnn_pool],  axis=1)
    stack_pool_prob.to_csv(os.path.join( name, "stack_pool_prob_cnn.csv"))
    stack_pool_pred.to_csv(os.path.join( name, "stack_pool_pred_cnn.csv"))
    
    stack_pool_prob = pd.read_csv(os.path.join( name, "stack_pool_prob_cnn.csv"), index_col=0)
    stack_pool_prob["y_prob_pool_average"] = stack_pool_prob.mean(axis=1)
    output_file = os.path.join(name, "y_prob_pool_average_cnn.csv")
    stack_pool_prob[["y_prob_pool_average"]].to_csv(output_file)
    x_pool_prob = pd.read_csv(os.path.join( name, "y_prob_pool_average_cnn.csv"), index_col=0)
    y_pred_pool_average, metrics_pool_average = y_prediction_average(x_pool_prob, y_pool, "y_pred_pool_average")
    y_pred_pool_average.to_csv(os.path.join( name, "y_pred_pool_average_cnn.csv"))
    
    metric_pool= pd.concat([y1_metric_cnn_pool,
                        y2_metric_cnn_pool,
                        y3_metric_cnn_pool,
                        y4_metric_cnn_pool,
                        y5_metric_cnn_pool,
                        y6_metric_cnn_pool, metrics_pool_average],  axis=0)
    
    metric_pool.to_csv(os.path.join( name, "metric_pool_cnn.csv"))

def main():
    for name in ['margin1/meta_pool_cnn']:
        print("#"*100) 
        print(name)
        meta_cnn_train(name, model_file='margin1/meta_average_cnn')
        print("finish train ", name)

if __name__ == "__main__":
    main()