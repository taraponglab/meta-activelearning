import pandas as pd
import numpy as np
import os
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Reshape
from keras.layers import Input, Dense, Attention
from joblib import dump
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

""" 
1. Load x (molecular features) and y (labels)
2. Train baseline models (CNN, BiLSTM, Attention)
3. Evaluate performance of all train and test
""" 

def cnn_model(fingerprint_length):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, input_shape=(fingerprint_length,1), activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def bilstm_model(fingerprint_length):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(fingerprint_length, 1))))
    model.add(Bidirectional(LSTM(128, return_sequences=False))) # return_sequences=False so the output is 2D (batch, units)
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def attention(fingerprint_length):
    input_layer = Input(shape=(fingerprint_length,))
    dense_layer = Dense(64, activation='relu')(input_layer)
    reshape_layer = Reshape((1, 64))(dense_layer)                               # Reshape layer to for attention
    attention_layer = Attention(use_scale=True)([reshape_layer, reshape_layer]) # Attention mechanism layer
    attention_output = Reshape((64,))(attention_layer)
    output_layer = Dense(1, activation='sigmoid')(attention_output)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

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

from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, balanced_accuracy_score, precision_score, average_precision_score, confusion_matrix

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob > threshold).astype(int).reshape(-1)

    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    bcc = balanced_accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    av_pre = average_precision_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spc = tn / (tn + fp)

    metrics = {
        "Accuracy": acc,
        "Sensitivity": sen,
        "Specificity": spc,
        "MCC": mcc,
        "F1 Score": f1,
        "AUC": auc,
        "Balanced Accuracy": bcc,
        "Precision": pre,
        "Average Precision": av_pre
    }
    return metrics


def get_oof_predictions(base_model_fn, x, y, n_splits=5, reshape_for_model=None, model_save_prefix=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(x), 1))  # shape: (num_samples, 1)

    for fold, (train_index, val_index) in enumerate(skf.split(x, y)):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        if reshape_for_model:
            x_train = reshape_for_model(x_train)
            x_val = reshape_for_model(x_val)

        model = base_model_fn(x_train.shape[1])
        model.fit(x_train, y_train, epochs=20, batch_size=32)
        oof_preds[val_index] = model.predict(x_val)
        
        # Save the model
        if model_save_prefix:
            model_save_path = f"{model_save_prefix}_fold{fold+1}.keras"
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")

    return oof_preds

def stacked_class(name, test_folder="data/test"):

    xat_train = pd.read_csv(os.path.join(name,  'AD2D.csv'     ), index_col=0)
    xes_train = pd.read_csv(os.path.join(name,  'EState.csv'   ), index_col=0)
    xke_train = pd.read_csv(os.path.join(name,  'KRFP.csv'     ), index_col=0)
    xpc_train = pd.read_csv(os.path.join(name,  'PubChem.csv'  ), index_col=0)
    xss_train = pd.read_csv(os.path.join(name,  'SubFP.csv'    ), index_col=0)
    xcd_train = pd.read_csv(os.path.join(name,  'CDKGraph.csv' ), index_col=0)
    xcn_train = pd.read_csv(os.path.join(name,  'CDK.csv'      ), index_col=0)
    xkc_train = pd.read_csv(os.path.join(name,  'KRFPC.csv'    ), index_col=0)
    xce_train = pd.read_csv(os.path.join(name,  'CDKExt.csv'   ), index_col=0)
    xsc_train = pd.read_csv(os.path.join(name,  'SubFPC.csv'   ), index_col=0)
    xac_train = pd.read_csv(os.path.join(name,  'AP2DC.csv'    ), index_col=0)
    xma_train = pd.read_csv(os.path.join(name,  'MACCS.csv'    ), index_col=0)
    y_train   = pd.read_csv(os.path.join(name,  "y_train.csv"  ), index_col=0)
    
    xat_train_np = np.array(xat_train)
    xes_train_np = np.array(xes_train)
    xke_train_np = np.array(xke_train)
    xpc_train_np = np.array(xpc_train)
    xss_train_np = np.array(xss_train)
    xcd_train_np = np.array(xcd_train)
    xcn_train_np = np.array(xcn_train)
    xkc_train_np = np.array(xkc_train)
    xce_train_np = np.array(xce_train)
    xsc_train_np = np.array(xsc_train)
    xac_train_np = np.array(xac_train)
    xma_train_np = np.array(xma_train)
    y_train = np.array(y_train)

    metrics_list = []

    def add_metrics(model_name, fingerprint_name, y_true, y_pred_prob):
        metrics = compute_metrics(y_true.ravel(), y_pred_prob.ravel())
        metrics["Model"] = model_name
        metrics["Fingerprint"] = fingerprint_name
        metrics_list.append(metrics)

    # Train CNN models
    yat_oof_cnn = get_oof_predictions(cnn_model, xat_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/cnn_AD2D")
    yes_oof_cnn = get_oof_predictions(cnn_model, xes_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/cnn_EState")
    yke_oof_cnn = get_oof_predictions(cnn_model, xke_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/cnn_KRFP")
    ypc_oof_cnn = get_oof_predictions(cnn_model, xpc_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/cnn_PubChem")
    yss_oof_cnn = get_oof_predictions(cnn_model, xss_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/cnn_SubFP")
    ycd_oof_cnn = get_oof_predictions(cnn_model, xcd_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/cnn_CDKGraph")
    ycn_oof_cnn = get_oof_predictions(cnn_model, xcn_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/cnn_CDK")
    ykc_oof_cnn = get_oof_predictions(cnn_model, xkc_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/cnn_KRFPC")
    yce_oof_cnn = get_oof_predictions(cnn_model, xce_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/cnn_CDKExt")
    ysc_oof_cnn = get_oof_predictions(cnn_model, xsc_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/cnn_SubFPC")
    yac_oof_cnn = get_oof_predictions(cnn_model, xac_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/cnn_AP2DC")
    yma_oof_cnn = get_oof_predictions(cnn_model, xma_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/cnn_MACCS")
    
    # Add metrics for CNN
    add_metrics("CNN", "AD2D", y_train, yat_oof_cnn)
    add_metrics("CNN", "EState", y_train, yes_oof_cnn)
    add_metrics("CNN", "KRFP", y_train, yke_oof_cnn)
    add_metrics("CNN", "PubChem", y_train, ypc_oof_cnn)
    add_metrics("CNN", "SubFP", y_train, yss_oof_cnn)
    add_metrics("CNN", "CDKGraph", y_train, ycd_oof_cnn)
    add_metrics("CNN", "CDK", y_train, ycn_oof_cnn)
    add_metrics("CNN", "KRFPC", y_train, ykc_oof_cnn)
    add_metrics("CNN", "CDKExt", y_train, yce_oof_cnn)
    add_metrics("CNN", "SubFPC", y_train, ysc_oof_cnn)
    add_metrics("CNN", "AP2DC", y_train, yac_oof_cnn)
    add_metrics("CNN", "MACCS", y_train, yma_oof_cnn)

    # Reshaping the training data for BiLSTM
    xat_train_np_bilstm = xat_train_np.reshape((-1, 1, xat_train_np.shape[1]))
    xes_train_np_bilstm = xes_train_np.reshape((-1, 1, xes_train_np.shape[1]))
    xke_train_np_bilstm = xke_train_np.reshape((-1, 1, xke_train_np.shape[1]))
    xpc_train_np_bilstm = xpc_train_np.reshape((-1, 1, xpc_train_np.shape[1]))
    xss_train_np_bilstm = xss_train_np.reshape((-1, 1, xss_train_np.shape[1]))
    xcd_train_np_bilstm = xcd_train_np.reshape((-1, 1, xcd_train_np.shape[1]))
    xcn_train_np_bilstm = xcn_train_np.reshape((-1, 1, xcn_train_np.shape[1]))
    xkc_train_np_bilstm = xkc_train_np.reshape((-1, 1, xkc_train_np.shape[1]))
    xce_train_np_bilstm = xce_train_np.reshape((-1, 1, xce_train_np.shape[1]))
    xsc_train_np_bilstm = xsc_train_np.reshape((-1, 1, xsc_train_np.shape[1]))
    xac_train_np_bilstm = xac_train_np.reshape((-1, 1, xac_train_np.shape[1]))
    xma_train_np_bilstm = xma_train_np.reshape((-1, 1, xma_train_np.shape[1]))

    # Train BiLSTM models
    yat_oof_bilstm = get_oof_predictions(bilstm_model, xat_train_np_bilstm, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/bilstm_AD2D")
    yes_oof_bilstm = get_oof_predictions(bilstm_model, xes_train_np_bilstm, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/bilstm_EState")
    yke_oof_bilstm = get_oof_predictions(bilstm_model, xke_train_np_bilstm, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/bilstm_KRFP")
    ypc_oof_bilstm = get_oof_predictions(bilstm_model, xpc_train_np_bilstm, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/bilstm_PubChem")
    yss_oof_bilstm = get_oof_predictions(bilstm_model, xss_train_np_bilstm, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/bilstm_SubFP")
    ycd_oof_bilstm = get_oof_predictions(bilstm_model, xcd_train_np_bilstm, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/bilstm_CDKGraph")
    ycn_oof_bilstm = get_oof_predictions(bilstm_model, xcn_train_np_bilstm, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/bilstm_CDK")
    ykc_oof_bilstm = get_oof_predictions(bilstm_model, xkc_train_np_bilstm, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/bilstm_KRFPC")
    yce_oof_bilstm = get_oof_predictions(bilstm_model, xce_train_np_bilstm, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/bilstm_CDKExt")
    ysc_oof_bilstm = get_oof_predictions(bilstm_model, xsc_train_np_bilstm, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/bilstm_SubFPC")
    yac_oof_bilstm = get_oof_predictions(bilstm_model, xac_train_np_bilstm, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/bilstm_AP2DC")
    yma_oof_bilstm = get_oof_predictions(bilstm_model, xma_train_np_bilstm, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/bilstm_MACCS")

    # Add metrics for BiLSTM
    add_metrics("BiLSTM", "AD2D", y_train, yat_oof_bilstm)
    add_metrics("BiLSTM", "EState", y_train, yes_oof_bilstm)
    add_metrics("BiLSTM", "KRFP", y_train, yke_oof_bilstm)
    add_metrics("BiLSTM", "PubChem", y_train, ypc_oof_bilstm)
    add_metrics("BiLSTM", "SubFP", y_train, yss_oof_bilstm)
    add_metrics("BiLSTM", "CDKGraph", y_train, ycd_oof_bilstm)
    add_metrics("BiLSTM", "CDK", y_train, ycn_oof_bilstm)
    add_metrics("BiLSTM", "KRFPC", y_train, ykc_oof_bilstm)
    add_metrics("BiLSTM", "CDKExt", y_train, yce_oof_bilstm)
    add_metrics("BiLSTM", "SubFPC", y_train, ysc_oof_bilstm)
    add_metrics("BiLSTM", "AP2DC", y_train, yac_oof_bilstm)
    add_metrics("BiLSTM", "MACCS", y_train, yma_oof_bilstm)

    # Train attention models
    yat_oof_att = get_oof_predictions(attention, xat_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/att_AD2D")
    yes_oof_att = get_oof_predictions(attention, xes_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/att_EState")
    yke_oof_att = get_oof_predictions(attention, xke_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/att_KRFP")
    ypc_oof_att = get_oof_predictions(attention, xpc_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/att_PubChem")
    yss_oof_att = get_oof_predictions(attention, xss_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/att_SubFP")
    ycd_oof_att = get_oof_predictions(attention, xcd_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/att_CDKGraph")
    ycn_oof_att = get_oof_predictions(attention, xcn_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/att_CDK")
    ykc_oof_att = get_oof_predictions(attention, xkc_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/att_KRFPC")
    yce_oof_att = get_oof_predictions(attention, xce_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/att_CDKExt")
    ysc_oof_att = get_oof_predictions(attention, xsc_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/att_SubFPC")
    yac_oof_att = get_oof_predictions(attention, xac_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/att_AP2DC")
    yma_oof_att = get_oof_predictions(attention, xma_train_np, y_train.ravel(), n_splits=5, reshape_for_model=None, model_save_prefix= f"{name}/att_MACCS")

    # Add metrics for Attention
    add_metrics("Attention", "AD2D", y_train, yat_oof_att)
    add_metrics("Attention", "EState", y_train, yes_oof_att)
    add_metrics("Attention", "KRFP", y_train, yke_oof_att)
    add_metrics("Attention", "PubChem", y_train, ypc_oof_att)
    add_metrics("Attention", "SubFP", y_train, yss_oof_att)
    add_metrics("Attention", "CDKGraph", y_train, ycd_oof_att)
    add_metrics("Attention", "CDK", y_train, ycn_oof_att)
    add_metrics("Attention", "KRFPC", y_train, ykc_oof_att)
    add_metrics("Attention", "CDKExt", y_train, yce_oof_att)
    add_metrics("Attention", "SubFPC", y_train, ysc_oof_att)
    add_metrics("Attention", "AP2DC", y_train, yac_oof_att)
    add_metrics("Attention", "MACCS", y_train, yma_oof_att)

    # Save all metrics
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(name, "evaluation_metrics.csv"), index=False)


    # Convert predictions to DataFrames
    yat_oof_cnn_df = pd.DataFrame(yat_oof_cnn, index=xat_train.index, columns=["yat_oof_cnn"])
    yes_oof_cnn_df = pd.DataFrame(yes_oof_cnn, index=xes_train.index, columns=["yes_oof_cnn"])
    yke_oof_cnn_df = pd.DataFrame(yke_oof_cnn, index=xke_train.index, columns=["yke_oof_cnn"])
    ypc_oof_cnn_df = pd.DataFrame(ypc_oof_cnn, index=xpc_train.index, columns=["ypc_oof_cnn"])
    yss_oof_cnn_df = pd.DataFrame(yss_oof_cnn, index=xss_train.index, columns=["yss_oof_cnn"])
    ycd_oof_cnn_df = pd.DataFrame(ycd_oof_cnn, index=xcd_train.index, columns=["ycd_oof_cnn"])
    ycn_oof_cnn_df = pd.DataFrame(ycn_oof_cnn, index=xcn_train.index, columns=["ycn_oof_cnn"])
    ykc_oof_cnn_df = pd.DataFrame(ykc_oof_cnn, index=xkc_train.index, columns=["ykc_oof_cnn"])
    yce_oof_cnn_df = pd.DataFrame(yce_oof_cnn, index=xce_train.index, columns=["yce_oof_cnn"])
    ysc_oof_cnn_df = pd.DataFrame(ysc_oof_cnn, index=xsc_train.index, columns=["ysc_oof_cnn"])
    yac_oof_cnn_df = pd.DataFrame(yac_oof_cnn, index=xac_train.index, columns=["yac_oof_cnn"])
    yma_oof_cnn_df = pd.DataFrame(yma_oof_cnn, index=xma_train.index, columns=["yma_oof_cnn"])
    yat_oof_bilstm_df = pd.DataFrame(yat_oof_bilstm, index=xat_train.index, columns=["yat_oof_bilstm"])
    yes_oof_bilstm_df = pd.DataFrame(yes_oof_bilstm, index=xes_train.index, columns=["yes_oof_bilstm"])
    yke_oof_bilstm_df = pd.DataFrame(yke_oof_bilstm, index=xke_train.index, columns=["yke_oof_bilstm"])
    ypc_oof_bilstm_df = pd.DataFrame(ypc_oof_bilstm, index=xpc_train.index, columns=["ypc_oof_bilstm"])
    yss_oof_bilstm_df = pd.DataFrame(yss_oof_bilstm, index=xss_train.index, columns=["yss_oof_bilstm"])
    ycd_oof_bilstm_df = pd.DataFrame(ycd_oof_bilstm, index=xcd_train.index, columns=["ycd_oof_bilstm"])
    ycn_oof_bilstm_df = pd.DataFrame(ycn_oof_bilstm, index=xcn_train.index, columns=["ycn_oof_bilstm"])
    ykc_oof_bilstm_df = pd.DataFrame(ykc_oof_bilstm, index=xkc_train.index, columns=["ykc_oof_bilstm"])
    yce_oof_bilstm_df = pd.DataFrame(yce_oof_bilstm, index=xce_train.index, columns=["yce_oof_bilstm"])
    ysc_oof_bilstm_df = pd.DataFrame(ysc_oof_bilstm, index=xsc_train.index, columns=["ysc_oof_bilstm"])
    yac_oof_bilstm_df = pd.DataFrame(yac_oof_bilstm, index=xac_train.index, columns=["yac_oof_bilstm"])
    yma_oof_bilstm_df = pd.DataFrame(yma_oof_bilstm, index=xma_train.index, columns=["yma_oof_bilstm"])
    yat_oof_att_df = pd.DataFrame(yat_oof_att, index=xat_train.index, columns=["yat_oof_att"])
    yes_oof_att_df = pd.DataFrame(yes_oof_att, index=xes_train.index, columns=["yes_oof_att"])
    yke_oof_att_df = pd.DataFrame(yke_oof_att, index=xke_train.index, columns=["yke_oof_att"])
    ypc_oof_att_df = pd.DataFrame(ypc_oof_att, index=xpc_train.index, columns=["ypc_oof_att"])
    yss_oof_att_df = pd.DataFrame(yss_oof_att, index=xss_train.index, columns=["yss_oof_att"])
    ycd_oof_att_df = pd.DataFrame(ycd_oof_att, index=xcd_train.index, columns=["ycd_oof_att"])
    ycn_oof_att_df = pd.DataFrame(ycn_oof_att, index=xcn_train.index, columns=["ycn_oof_att"])
    ykc_oof_att_df = pd.DataFrame(ykc_oof_att, index=xkc_train.index, columns=["ykc_oof_att"])
    yce_oof_att_df = pd.DataFrame(yce_oof_att, index=xce_train.index, columns=["yce_oof_att"])
    ysc_oof_att_df = pd.DataFrame(ysc_oof_att, index=xsc_train.index, columns=["ysc_oof_att"])
    yac_oof_att_df = pd.DataFrame(yac_oof_att, index=xac_train.index, columns=["yac_oof_att"])
    yma_oof_att_df = pd.DataFrame(yma_oof_att, index=xma_train.index, columns=["yma_oof_att"])

    # Stack the predictive features
    stack_train_oof_all = pd.concat([yat_oof_cnn_df, yat_oof_bilstm_df, yat_oof_att_df,
                            yes_oof_cnn_df, yes_oof_bilstm_df, yes_oof_att_df,
                            yke_oof_cnn_df, yke_oof_bilstm_df, yke_oof_att_df,
                            ypc_oof_cnn_df, ypc_oof_bilstm_df, ypc_oof_att_df,
                            yss_oof_cnn_df, yss_oof_bilstm_df, yss_oof_att_df,
                            ycd_oof_cnn_df, ycd_oof_bilstm_df, ycd_oof_att_df,
                            ycn_oof_cnn_df, ycn_oof_bilstm_df, ycn_oof_att_df,
                            ykc_oof_cnn_df, ykc_oof_bilstm_df, ykc_oof_att_df,
                            yce_oof_cnn_df, yce_oof_bilstm_df, yce_oof_att_df,
                            ysc_oof_cnn_df, ysc_oof_bilstm_df, ysc_oof_att_df,
                            yac_oof_cnn_df, yac_oof_bilstm_df, yac_oof_att_df,
                            yma_oof_cnn_df, yma_oof_bilstm_df, yma_oof_att_df],  axis=1)

    stack_train_oof_all.to_csv (os.path.join(name, "all_stacked_train_oof.csv"))

def main():
    input_folders = input("Enter folder names separated by commas (e.g., subset1/train_1,subset1/train_2): ")
    folder_list = [name.strip() for name in input_folders.split(",") if name.strip()]
    for name in folder_list:
        print("#"*100) 
        print(name)
        y_train  = pd.read_csv(os.path.join(name, "y_train.csv"), index_col=0)
        print(y_train)
        stacked_class(name)
        print("Finish training model ", name)

if __name__ == "__main__":
    main() 