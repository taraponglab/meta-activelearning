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
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
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

def stacked_class(name):

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
    
    xat_test = pd.read_csv(os.path.join("test",  'AD2D.csv'     ), index_col=0)
    xes_test = pd.read_csv(os.path.join("test",  'EState.csv'   ), index_col=0)
    xke_test = pd.read_csv(os.path.join("test",  'KRFP.csv'     ), index_col=0)
    xpc_test = pd.read_csv(os.path.join("test",  'PubChem.csv'  ), index_col=0)
    xss_test = pd.read_csv(os.path.join("test",  'SubFP.csv'    ), index_col=0)
    xcd_test = pd.read_csv(os.path.join("test",  'CDKGraph.csv' ), index_col=0)
    xcn_test = pd.read_csv(os.path.join("test",  'CDK.csv'      ), index_col=0)
    xkc_test = pd.read_csv(os.path.join("test",  'KRFPC.csv'    ), index_col=0)
    xce_test = pd.read_csv(os.path.join("test",  'CDKExt.csv'   ), index_col=0)
    xsc_test = pd.read_csv(os.path.join("test",  'SubFPC.csv'   ), index_col=0)
    xac_test = pd.read_csv(os.path.join("test",  'AP2DC.csv'    ), index_col=0)
    xma_test = pd.read_csv(os.path.join("test",  'MACCS.csv'    ), index_col=0)
    y_test   = pd.read_csv(os.path.join("test",  "y_test.csv"   ), index_col=0)
    
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
    
    xat_test_np = np.array(xat_test)
    xes_test_np = np.array(xes_test)
    xke_test_np = np.array(xke_test)
    xpc_test_np = np.array(xpc_test)
    xss_test_np = np.array(xss_test)
    xcd_test_np = np.array(xcd_test)
    xcn_test_np = np.array(xcn_test)
    xkc_test_np = np.array(xkc_test)
    xce_test_np = np.array(xce_test)
    xsc_test_np = np.array(xsc_test)
    xac_test_np = np.array(xac_test)
    xma_test_np = np.array(xma_test)
    y_test = np.array(y_test)


    xat_train_split, xat_val_split, yat_train_split, yat_val_split = train_test_split(xat_train_np, y_train, test_size=0.2, random_state=42)
    xes_train_split, xes_val_split, yes_train_split, yes_val_split = train_test_split(xes_train_np, y_train, test_size=0.2, random_state=42)
    xke_train_split, xke_val_split, yke_train_split, yke_val_split = train_test_split(xke_train_np, y_train, test_size=0.2, random_state=42)
    xpc_train_split, xpc_val_split, ypc_train_split, ypc_val_split = train_test_split(xpc_train_np, y_train, test_size=0.2, random_state=42)
    xss_train_split, xss_val_split, yss_train_split, yss_val_split = train_test_split(xss_train_np, y_train, test_size=0.2, random_state=42)
    xcd_train_split, xcd_val_split, ycd_train_split, ycd_val_split = train_test_split(xcd_train_np, y_train, test_size=0.2, random_state=42)
    xcn_train_split, xcn_val_split, ycn_train_split, ycn_val_split = train_test_split(xcn_train_np, y_train, test_size=0.2, random_state=42)
    xkc_train_split, xkc_val_split, ykc_train_split, ykc_val_split = train_test_split(xkc_train_np, y_train, test_size=0.2, random_state=42)
    xce_train_split, xce_val_split, yce_train_split, yce_val_split = train_test_split(xce_train_np, y_train, test_size=0.2, random_state=42)
    xsc_train_split, xsc_val_split, ysc_train_split, ysc_val_split = train_test_split(xsc_train_np, y_train, test_size=0.2, random_state=42)
    xac_train_split, xac_val_split, yac_train_split, yac_val_split = train_test_split(xac_train_np, y_train, test_size=0.2, random_state=42)
    xma_train_split, xma_val_split, yma_train_split, yma_val_split = train_test_split(xma_train_np, y_train, test_size=0.2, random_state=42)

    
    # Train CNN models
    baseline_model_cnn_at = cnn_model(fingerprint_length=xat_train_split.shape[1])
    baseline_model_cnn_es = cnn_model(fingerprint_length=xes_train_split.shape[1])
    baseline_model_cnn_ke = cnn_model(fingerprint_length=xke_train_split.shape[1])
    baseline_model_cnn_pc = cnn_model(fingerprint_length=xpc_train_split.shape[1])
    baseline_model_cnn_ss = cnn_model(fingerprint_length=xss_train_split.shape[1])
    baseline_model_cnn_cd = cnn_model(fingerprint_length=xcd_train_split.shape[1])
    baseline_model_cnn_cn = cnn_model(fingerprint_length=xcn_train_split.shape[1])
    baseline_model_cnn_kc = cnn_model(fingerprint_length=xkc_train_split.shape[1])
    baseline_model_cnn_ce = cnn_model(fingerprint_length=xce_train_split.shape[1])
    baseline_model_cnn_sc = cnn_model(fingerprint_length=xsc_train_split.shape[1])
    baseline_model_cnn_ac = cnn_model(fingerprint_length=xac_train_split.shape[1])
    baseline_model_cnn_ma = cnn_model(fingerprint_length=xma_train_split.shape[1])
    baseline_model_cnn_at.fit(xat_train_split, yat_train_split, validation_data=(xat_val_split, yat_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_es.fit(xes_train_split, yes_train_split, validation_data=(xes_val_split, yes_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_ke.fit(xke_train_split, yke_train_split, validation_data=(xke_val_split, yke_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_pc.fit(xpc_train_split, ypc_train_split, validation_data=(xpc_val_split, ypc_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_ss.fit(xss_train_split, yss_train_split, validation_data=(xss_val_split, yss_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_cd.fit(xcd_train_split, ycd_train_split, validation_data=(xcd_val_split, ycd_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_cn.fit(xcn_train_split, ycn_train_split, validation_data=(xcn_val_split, ycn_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_kc.fit(xkc_train_split, ykc_train_split, validation_data=(xkc_val_split, ykc_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_ce.fit(xce_train_split, yce_train_split, validation_data=(xce_val_split, yce_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_sc.fit(xsc_train_split, ysc_train_split, validation_data=(xsc_val_split, ysc_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_ac.fit(xac_train_split, yac_train_split, validation_data=(xac_val_split, yac_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_ma.fit(xma_train_split, yma_train_split, validation_data=(xma_val_split, yma_val_split), epochs=20, batch_size=32)
    
    # Save the trained models
    baseline_model_cnn_at.save(os.path.join(name, "baseline_model_cnn_at.keras"))
    baseline_model_cnn_es.save(os.path.join(name, "baseline_model_cnn_es.keras"))
    baseline_model_cnn_ke.save(os.path.join(name, "baseline_model_cnn_ke.keras"))
    baseline_model_cnn_pc.save(os.path.join(name, "baseline_model_cnn_pc.keras"))
    baseline_model_cnn_ss.save(os.path.join(name, "baseline_model_cnn_ss.keras"))
    baseline_model_cnn_cd.save(os.path.join(name, "baseline_model_cnn_cd.keras"))
    baseline_model_cnn_cn.save(os.path.join(name, "baseline_model_cnn_cn.keras"))
    baseline_model_cnn_kc.save(os.path.join(name, "baseline_model_cnn_kc.keras"))
    baseline_model_cnn_ce.save(os.path.join(name, "baseline_model_cnn_ce.keras"))
    baseline_model_cnn_sc.save(os.path.join(name, "baseline_model_cnn_sc.keras"))
    baseline_model_cnn_ac.save(os.path.join(name, "baseline_model_cnn_ac.keras"))
    baseline_model_cnn_ma.save(os.path.join(name, "baseline_model_cnn_ma.keras"))
    # Predict with CNN models
    yat_prob_cnn_train, yat_pred_cnn_train, yat_metric_cnn_train = y_prediction(baseline_model_cnn_at, xat_train_np, xat_train, y_train, "yat_pred_cnn")
    yes_prob_cnn_train, yes_pred_cnn_train, yes_metric_cnn_train = y_prediction(baseline_model_cnn_es, xes_train_np, xes_train, y_train, "yes_pred_cnn")
    yke_prob_cnn_train, yke_pred_cnn_train, yke_metric_cnn_train = y_prediction(baseline_model_cnn_ke, xke_train_np, xke_train, y_train, "yke_pred_cnn")
    ypc_prob_cnn_train, ypc_pred_cnn_train, ypc_metric_cnn_train = y_prediction(baseline_model_cnn_pc, xpc_train_np, xpc_train, y_train, "ypc_pred_cnn")
    yss_prob_cnn_train, yss_pred_cnn_train, yss_metric_cnn_train = y_prediction(baseline_model_cnn_ss, xss_train_np, xss_train, y_train, "yss_pred_cnn")
    ycd_prob_cnn_train, ycd_pred_cnn_train, ycd_metric_cnn_train = y_prediction(baseline_model_cnn_cd, xcd_train_np, xcd_train, y_train, "ycd_pred_cnn")
    ycn_prob_cnn_train, ycn_pred_cnn_train, ycn_metric_cnn_train = y_prediction(baseline_model_cnn_cn, xcn_train_np, xcn_train, y_train, "ycn_pred_cnn")
    ykc_prob_cnn_train, ykc_pred_cnn_train, ykc_metric_cnn_train = y_prediction(baseline_model_cnn_kc, xkc_train_np, xkc_train, y_train, "ykc_pred_cnn")
    yce_prob_cnn_train, yce_pred_cnn_train, yce_metric_cnn_train = y_prediction(baseline_model_cnn_ce, xce_train_np, xce_train, y_train, "yce_pred_cnn")
    ysc_prob_cnn_train, ysc_pred_cnn_train, ysc_metric_cnn_train = y_prediction(baseline_model_cnn_sc, xsc_train_np, xsc_train, y_train, "ysc_pred_cnn")
    yac_prob_cnn_train, yac_pred_cnn_train, yac_metric_cnn_train = y_prediction(baseline_model_cnn_ac, xac_train_np, xac_train, y_train, "yac_pred_cnn")
    yma_prob_cnn_train, yma_pred_cnn_train, yma_metric_cnn_train = y_prediction(baseline_model_cnn_ma, xma_train_np, xma_train, y_train, "yma_pred_cnn")
    yat_prob_cnn_test, yat_pred_cnn_test, yat_metric_cnn_test   =  y_prediction(baseline_model_cnn_at, xat_test_np, xat_test, y_test, "yat_pred_cnn")
    yes_prob_cnn_test, yes_pred_cnn_test, yes_metric_cnn_test   =  y_prediction(baseline_model_cnn_es, xes_test_np, xes_test, y_test, "yes_pred_cnn")
    yke_prob_cnn_test, yke_pred_cnn_test, yke_metric_cnn_test   =  y_prediction(baseline_model_cnn_ke, xke_test_np, xke_test, y_test, "yke_pred_cnn")
    ypc_prob_cnn_test, ypc_pred_cnn_test, ypc_metric_cnn_test   =  y_prediction(baseline_model_cnn_pc, xpc_test_np, xpc_test, y_test, "ypc_pred_cnn")
    yss_prob_cnn_test, yss_pred_cnn_test, yss_metric_cnn_test   =  y_prediction(baseline_model_cnn_ss, xss_test_np, xss_test, y_test, "yss_pred_cnn")
    ycd_prob_cnn_test, ycd_pred_cnn_test, ycd_metric_cnn_test   =  y_prediction(baseline_model_cnn_cd, xcd_test_np, xcd_test, y_test, "ycd_pred_cnn")
    ycn_prob_cnn_test, ycn_pred_cnn_test, ycn_metric_cnn_test   =  y_prediction(baseline_model_cnn_cn, xcn_test_np, xcn_test, y_test, "ycn_pred_cnn")
    ykc_prob_cnn_test, ykc_pred_cnn_test, ykc_metric_cnn_test   =  y_prediction(baseline_model_cnn_kc, xkc_test_np, xkc_test, y_test, "ykc_pred_cnn")
    yce_prob_cnn_test, yce_pred_cnn_test, yce_metric_cnn_test   =  y_prediction(baseline_model_cnn_ce, xce_test_np, xce_test, y_test, "yce_pred_cnn")
    ysc_prob_cnn_test, ysc_pred_cnn_test, ysc_metric_cnn_test   =  y_prediction(baseline_model_cnn_sc, xsc_test_np, xsc_test, y_test, "ysc_pred_cnn")
    yac_prob_cnn_test, yac_pred_cnn_test, yac_metric_cnn_test   =  y_prediction(baseline_model_cnn_ac, xac_test_np, xac_test, y_test, "yac_pred_cnn")
    yma_prob_cnn_test, yma_pred_cnn_test, yma_metric_cnn_test   =  y_prediction(baseline_model_cnn_ma, xma_test_np, xma_test, y_test, "yma_pred_cnn")
    
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

    # Reshaping the test data  for BiLSTM
    xat_test_np_bilstm = xat_test_np.reshape((-1, 1, xat_test_np.shape[1]))
    xes_test_np_bilstm = xes_test_np.reshape((-1, 1, xes_test_np.shape[1]))
    xke_test_np_bilstm = xke_test_np.reshape((-1, 1, xke_test_np.shape[1]))
    xpc_test_np_bilstm = xpc_test_np.reshape((-1, 1, xpc_test_np.shape[1]))
    xss_test_np_bilstm = xss_test_np.reshape((-1, 1, xss_test_np.shape[1]))
    xcd_test_np_bilstm = xcd_test_np.reshape((-1, 1, xcd_test_np.shape[1]))
    xcn_test_np_bilstm = xcn_test_np.reshape((-1, 1, xcn_test_np.shape[1]))
    xkc_test_np_bilstm = xkc_test_np.reshape((-1, 1, xkc_test_np.shape[1]))
    xce_test_np_bilstm = xce_test_np.reshape((-1, 1, xce_test_np.shape[1]))
    xsc_test_np_bilstm = xsc_test_np.reshape((-1, 1, xsc_test_np.shape[1]))
    xac_test_np_bilstm = xac_test_np.reshape((-1, 1, xac_test_np.shape[1]))
    xma_test_np_bilstm = xma_test_np.reshape((-1, 1, xma_test_np.shape[1]))
    
    xat_train_split_bilstm, xat_val_split_bilstm, yat_train_split_bilstm, yat_val_split_bilstm = train_test_split(xat_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xes_train_split_bilstm, xes_val_split_bilstm, yes_train_split_bilstm, yes_val_split_bilstm = train_test_split(xes_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xke_train_split_bilstm, xke_val_split_bilstm, yke_train_split_bilstm, yke_val_split_bilstm = train_test_split(xke_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xpc_train_split_bilstm, xpc_val_split_bilstm, ypc_train_split_bilstm, ypc_val_split_bilstm = train_test_split(xpc_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xss_train_split_bilstm, xss_val_split_bilstm, yss_train_split_bilstm, yss_val_split_bilstm = train_test_split(xss_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xcd_train_split_bilstm, xcd_val_split_bilstm, ycd_train_split_bilstm, ycd_val_split_bilstm = train_test_split(xcd_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xcn_train_split_bilstm, xcn_val_split_bilstm, ycn_train_split_bilstm, ycn_val_split_bilstm = train_test_split(xcn_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xkc_train_split_bilstm, xkc_val_split_bilstm, ykc_train_split_bilstm, ykc_val_split_bilstm = train_test_split(xkc_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xce_train_split_bilstm, xce_val_split_bilstm, yce_train_split_bilstm, yce_val_split_bilstm = train_test_split(xce_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xsc_train_split_bilstm, xsc_val_split_bilstm, ysc_train_split_bilstm, ysc_val_split_bilstm = train_test_split(xsc_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xac_train_split_bilstm, xac_val_split_bilstm, yac_train_split_bilstm, yac_val_split_bilstm = train_test_split(xac_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xma_train_split_bilstm, xma_val_split_bilstm, yma_train_split_bilstm, yma_val_split_bilstm = train_test_split(xma_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    
    # Train BiLSTM models
    baseline_model_bilstm_at = bilstm_model(fingerprint_length=xat_train.shape[1])
    baseline_model_bilstm_es = bilstm_model(fingerprint_length=xes_train.shape[1])
    baseline_model_bilstm_ke = bilstm_model(fingerprint_length=xke_train.shape[1])
    baseline_model_bilstm_pc = bilstm_model(fingerprint_length=xpc_train.shape[1])
    baseline_model_bilstm_ss = bilstm_model(fingerprint_length=xss_train.shape[1])
    baseline_model_bilstm_cd = bilstm_model(fingerprint_length=xcd_train.shape[1])
    baseline_model_bilstm_cn = bilstm_model(fingerprint_length=xcn_train.shape[1])
    baseline_model_bilstm_kc = bilstm_model(fingerprint_length=xkc_train.shape[1])
    baseline_model_bilstm_ce = bilstm_model(fingerprint_length=xce_train.shape[1])
    baseline_model_bilstm_sc = bilstm_model(fingerprint_length=xsc_train.shape[1])
    baseline_model_bilstm_ac = bilstm_model(fingerprint_length=xac_train.shape[1])
    baseline_model_bilstm_ma = bilstm_model(fingerprint_length=xma_train.shape[1])
    baseline_model_bilstm_at.fit(xat_train_split_bilstm, yat_train_split_bilstm, validation_data=(xat_val_split_bilstm, yat_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_es.fit(xes_train_split_bilstm, yes_train_split_bilstm, validation_data=(xes_val_split_bilstm, yes_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_ke.fit(xke_train_split_bilstm, yke_train_split_bilstm, validation_data=(xke_val_split_bilstm, yke_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_pc.fit(xpc_train_split_bilstm, ypc_train_split_bilstm, validation_data=(xpc_val_split_bilstm, ypc_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_ss.fit(xss_train_split_bilstm, yss_train_split_bilstm, validation_data=(xss_val_split_bilstm, yss_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_cd.fit(xcd_train_split_bilstm, ycd_train_split_bilstm, validation_data=(xcd_val_split_bilstm, ycd_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_cn.fit(xcn_train_split_bilstm, ycn_train_split_bilstm, validation_data=(xcn_val_split_bilstm, ycn_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_kc.fit(xkc_train_split_bilstm, ykc_train_split_bilstm, validation_data=(xkc_val_split_bilstm, ykc_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_ce.fit(xce_train_split_bilstm, yce_train_split_bilstm, validation_data=(xce_val_split_bilstm, yce_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_sc.fit(xsc_train_split_bilstm, ysc_train_split_bilstm, validation_data=(xsc_val_split_bilstm, ysc_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_ac.fit(xac_train_split_bilstm, yac_train_split_bilstm, validation_data=(xac_val_split_bilstm, yac_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_ma.fit(xma_train_split_bilstm, yma_train_split_bilstm, validation_data=(xma_val_split_bilstm, yma_val_split_bilstm), epochs=20, batch_size=32)
    
    # Save the trained models
    baseline_model_bilstm_at.save(os.path.join(name, "baseline_model_bilstm_at.keras"))
    baseline_model_bilstm_es.save(os.path.join(name, "baseline_model_bilstm_es.keras"))
    baseline_model_bilstm_ke.save(os.path.join(name, "baseline_model_bilstm_ke.keras"))
    baseline_model_bilstm_pc.save(os.path.join(name, "baseline_model_bilstm_pc.keras"))
    baseline_model_bilstm_ss.save(os.path.join(name, "baseline_model_bilstm_ss.keras"))
    baseline_model_bilstm_cd.save(os.path.join(name, "baseline_model_bilstm_cd.keras"))
    baseline_model_bilstm_cn.save(os.path.join(name, "baseline_model_bilstm_cn.keras"))
    baseline_model_bilstm_kc.save(os.path.join(name, "baseline_model_bilstm_kc.keras"))
    baseline_model_bilstm_ce.save(os.path.join(name, "baseline_model_bilstm_ce.keras"))
    baseline_model_bilstm_sc.save(os.path.join(name, "baseline_model_bilstm_sc.keras"))
    baseline_model_bilstm_ac.save(os.path.join(name, "baseline_model_bilstm_ac.keras"))
    baseline_model_bilstm_ma.save(os.path.join(name, "baseline_model_bilstm_ma.keras"))
    
    # Predict with BiLSTM models
    yat_prob_bilstm_train, yat_pred_bilstm_train, yat_metric_bilstm_train = y_prediction(   baseline_model_bilstm_at, xat_train_np_bilstm, xat_train, y_train, "yat_pred_bilstm")
    yes_prob_bilstm_train, yes_pred_bilstm_train, yes_metric_bilstm_train = y_prediction(   baseline_model_bilstm_es, xes_train_np_bilstm, xes_train, y_train, "yes_pred_bilstm")
    yke_prob_bilstm_train, yke_pred_bilstm_train, yke_metric_bilstm_train = y_prediction(   baseline_model_bilstm_ke, xke_train_np_bilstm, xke_train, y_train, "yke_pred_bilstm")
    ypc_prob_bilstm_train, ypc_pred_bilstm_train, ypc_metric_bilstm_train = y_prediction(   baseline_model_bilstm_pc, xpc_train_np_bilstm, xpc_train, y_train, "ypc_pred_bilstm")
    yss_prob_bilstm_train, yss_pred_bilstm_train, yss_metric_bilstm_train = y_prediction(   baseline_model_bilstm_ss, xss_train_np_bilstm, xss_train, y_train, "yss_pred_bilstm")
    ycd_prob_bilstm_train, ycd_pred_bilstm_train, ycd_metric_bilstm_train = y_prediction(   baseline_model_bilstm_cd, xcd_train_np_bilstm, xcd_train, y_train, "ycd_pred_bilstm")
    ycn_prob_bilstm_train, ycn_pred_bilstm_train, ycn_metric_bilstm_train = y_prediction(   baseline_model_bilstm_cn, xcn_train_np_bilstm, xcn_train, y_train, "ycn_pred_bilstm")
    ykc_prob_bilstm_train, ykc_pred_bilstm_train, ykc_metric_bilstm_train = y_prediction(   baseline_model_bilstm_kc, xkc_train_np_bilstm, xkc_train, y_train, "ykc_pred_bilstm")
    yce_prob_bilstm_train, yce_pred_bilstm_train, yce_metric_bilstm_train = y_prediction(   baseline_model_bilstm_ce, xce_train_np_bilstm, xce_train, y_train, "yce_pred_bilstm")
    ysc_prob_bilstm_train, ysc_pred_bilstm_train, ysc_metric_bilstm_train = y_prediction(   baseline_model_bilstm_sc, xsc_train_np_bilstm, xsc_train, y_train, "ysc_pred_bilstm")
    yac_prob_bilstm_train, yac_pred_bilstm_train, yac_metric_bilstm_train = y_prediction(   baseline_model_bilstm_ac, xac_train_np_bilstm, xac_train, y_train, "yac_pred_bilstm")
    yma_prob_bilstm_train, yma_pred_bilstm_train, yma_metric_bilstm_train = y_prediction(   baseline_model_bilstm_ma, xma_train_np_bilstm, xma_train, y_train, "yma_pred_bilstm")
    yat_prob_bilstm_test, yat_pred_bilstm_test,  yat_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_at, xat_test_np_bilstm, xat_test, y_test,   "yat_pred_bilstm")
    yes_prob_bilstm_test, yes_pred_bilstm_test,  yes_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_es, xes_test_np_bilstm, xes_test, y_test,   "yes_pred_bilstm")
    yke_prob_bilstm_test, yke_pred_bilstm_test,  yke_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_ke, xke_test_np_bilstm, xke_test, y_test,   "yke_pred_bilstm")
    ypc_prob_bilstm_test, ypc_pred_bilstm_test,  ypc_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_pc, xpc_test_np_bilstm, xpc_test, y_test,   "ypc_pred_bilstm")
    yss_prob_bilstm_test, yss_pred_bilstm_test,  yss_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_ss, xss_test_np_bilstm, xss_test, y_test,   "yss_pred_bilstm")
    ycd_prob_bilstm_test, ycd_pred_bilstm_test,  ycd_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_cd, xcd_test_np_bilstm, xcd_test, y_test,   "ycd_pred_bilstm")
    ycn_prob_bilstm_test, ycn_pred_bilstm_test,  ycn_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_cn, xcn_test_np_bilstm, xcn_test, y_test,   "ycn_pred_bilstm")
    ykc_prob_bilstm_test, ykc_pred_bilstm_test,  ykc_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_kc, xkc_test_np_bilstm, xkc_test, y_test,   "ykc_pred_bilstm")
    yce_prob_bilstm_test, yce_pred_bilstm_test,  yce_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_ce, xce_test_np_bilstm, xce_test, y_test,   "yce_pred_bilstm")
    ysc_prob_bilstm_test, ysc_pred_bilstm_test,  ysc_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_sc, xsc_test_np_bilstm, xsc_test, y_test,   "ysc_pred_bilstm")
    yac_prob_bilstm_test, yac_pred_bilstm_test,  yac_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_ac, xac_test_np_bilstm, xac_test, y_test,   "yac_pred_bilstm")
    yma_prob_bilstm_test, yma_pred_bilstm_test,  yma_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_ma, xma_test_np_bilstm, xma_test, y_test,   "yma_pred_bilstm")
    
    # Train attention models
    baseline_model_att_at = attention(fingerprint_length=xat_train.shape[1])
    baseline_model_att_es = attention(fingerprint_length=xes_train.shape[1])
    baseline_model_att_ke = attention(fingerprint_length=xke_train.shape[1])
    baseline_model_att_pc = attention(fingerprint_length=xpc_train.shape[1])
    baseline_model_att_ss = attention(fingerprint_length=xss_train.shape[1])
    baseline_model_att_cd = attention(fingerprint_length=xcd_train.shape[1])
    baseline_model_att_cn = attention(fingerprint_length=xcn_train.shape[1])
    baseline_model_att_kc = attention(fingerprint_length=xkc_train.shape[1])
    baseline_model_att_ce = attention(fingerprint_length=xce_train.shape[1])
    baseline_model_att_sc = attention(fingerprint_length=xsc_train.shape[1])
    baseline_model_att_ac = attention(fingerprint_length=xac_train.shape[1])
    baseline_model_att_ma = attention(fingerprint_length=xma_train.shape[1])
    baseline_model_att_at.fit(xat_train_split, yat_train_split, validation_data=(xat_val_split, yat_val_split), epochs=20, batch_size=32)
    baseline_model_att_es.fit(xes_train_split, yes_train_split, validation_data=(xes_val_split, yes_val_split), epochs=20, batch_size=32)
    baseline_model_att_ke.fit(xke_train_split, yke_train_split, validation_data=(xke_val_split, yke_val_split), epochs=20, batch_size=32)
    baseline_model_att_pc.fit(xpc_train_split, ypc_train_split, validation_data=(xpc_val_split, ypc_val_split), epochs=20, batch_size=32)
    baseline_model_att_ss.fit(xss_train_split, yss_train_split, validation_data=(xss_val_split, yss_val_split), epochs=20, batch_size=32)
    baseline_model_att_cd.fit(xcd_train_split, ycd_train_split, validation_data=(xcd_val_split, ycd_val_split), epochs=20, batch_size=32)
    baseline_model_att_cn.fit(xcn_train_split, ycn_train_split, validation_data=(xcn_val_split, ycn_val_split), epochs=20, batch_size=32)
    baseline_model_att_kc.fit(xkc_train_split, ykc_train_split, validation_data=(xkc_val_split, ykc_val_split), epochs=20, batch_size=32)
    baseline_model_att_ce.fit(xce_train_split, yce_train_split, validation_data=(xce_val_split, yce_val_split), epochs=20, batch_size=32)
    baseline_model_att_sc.fit(xsc_train_split, ysc_train_split, validation_data=(xsc_val_split, ysc_val_split), epochs=20, batch_size=32)
    baseline_model_att_ac.fit(xac_train_split, yac_train_split, validation_data=(xac_val_split, yac_val_split), epochs=20, batch_size=32)
    baseline_model_att_ma.fit(xma_train_split, yma_train_split, validation_data=(xma_val_split, yma_val_split), epochs=20, batch_size=32)
    
    # Save the trained models
    baseline_model_att_at.save(os.path.join(name, "baseline_model_att_at.keras"))
    baseline_model_att_es.save(os.path.join(name, "baseline_model_att_es.keras"))
    baseline_model_att_ke.save(os.path.join(name, "baseline_model_att_ke.keras"))
    baseline_model_att_pc.save(os.path.join(name, "baseline_model_att_pc.keras"))
    baseline_model_att_ss.save(os.path.join(name, "baseline_model_att_ss.keras"))
    baseline_model_att_cd.save(os.path.join(name, "baseline_model_att_cd.keras"))
    baseline_model_att_cn.save(os.path.join(name, "baseline_model_att_cn.keras"))
    baseline_model_att_kc.save(os.path.join(name, "baseline_model_att_kc.keras"))
    baseline_model_att_ce.save(os.path.join(name, "baseline_model_att_ce.keras"))
    baseline_model_att_sc.save(os.path.join(name, "baseline_model_att_sc.keras"))
    baseline_model_att_ac.save(os.path.join(name, "baseline_model_att_ac.keras"))
    baseline_model_att_ma.save(os.path.join(name, "baseline_model_att_ma.keras"))
    # Predict with attention models
    yat_prob_att_train, yat_pred_att_train, yat_metric_att_train = y_prediction(   baseline_model_att_at, xat_train_np, xat_train, y_train, "yat_pred_att")
    yes_prob_att_train, yes_pred_att_train, yes_metric_att_train = y_prediction(   baseline_model_att_es, xes_train_np, xes_train, y_train, "yes_pred_att")
    yke_prob_att_train, yke_pred_att_train, yke_metric_att_train = y_prediction(   baseline_model_att_ke, xke_train_np, xke_train, y_train, "yke_pred_att")
    ypc_prob_att_train, ypc_pred_att_train, ypc_metric_att_train = y_prediction(   baseline_model_att_pc, xpc_train_np, xpc_train, y_train, "ypc_pred_att")
    yss_prob_att_train, yss_pred_att_train, yss_metric_att_train = y_prediction(   baseline_model_att_ss, xss_train_np, xss_train, y_train, "yss_pred_att")
    ycd_prob_att_train, ycd_pred_att_train, ycd_metric_att_train = y_prediction(   baseline_model_att_cd, xcd_train_np, xcd_train, y_train, "ycd_pred_att")
    ycn_prob_att_train, ycn_pred_att_train, ycn_metric_att_train = y_prediction(   baseline_model_att_cn, xcn_train_np, xcn_train, y_train, "ycn_pred_att")
    ykc_prob_att_train, ykc_pred_att_train, ykc_metric_att_train = y_prediction(   baseline_model_att_kc, xkc_train_np, xkc_train, y_train, "ykc_pred_att")
    yce_prob_att_train, yce_pred_att_train, yce_metric_att_train = y_prediction(   baseline_model_att_ce, xce_train_np, xce_train, y_train, "yce_pred_att")
    ysc_prob_att_train, ysc_pred_att_train, ysc_metric_att_train = y_prediction(   baseline_model_att_sc, xsc_train_np, xsc_train, y_train, "ysc_pred_att")
    yac_prob_att_train, yac_pred_att_train, yac_metric_att_train = y_prediction(   baseline_model_att_ac, xac_train_np, xac_train, y_train, "yac_pred_att")
    yma_prob_att_train, yma_pred_att_train, yma_metric_att_train = y_prediction(   baseline_model_att_ma, xma_train_np, xma_train, y_train, "yma_pred_att")
    yat_prob_att_test, yat_pred_att_test,  yat_metric_att_test   = y_prediction(  baseline_model_att_at, xat_test_np, xat_test, y_test,   "yat_pred_att")
    yes_prob_att_test, yes_pred_att_test,  yes_metric_att_test   = y_prediction(  baseline_model_att_es, xes_test_np, xes_test, y_test,   "yes_pred_att")
    yke_prob_att_test, yke_pred_att_test,  yke_metric_att_test   = y_prediction(  baseline_model_att_ke, xke_test_np, xke_test, y_test,   "yke_pred_att")
    ypc_prob_att_test, ypc_pred_att_test,  ypc_metric_att_test   = y_prediction(  baseline_model_att_pc, xpc_test_np, xpc_test, y_test,   "ypc_pred_att")
    yss_prob_att_test, yss_pred_att_test,  yss_metric_att_test   = y_prediction(  baseline_model_att_ss, xss_test_np, xss_test, y_test,   "yss_pred_att")
    ycd_prob_att_test, ycd_pred_att_test,  ycd_metric_att_test   = y_prediction(  baseline_model_att_cd, xcd_test_np, xcd_test, y_test,   "ycd_pred_att")
    ycn_prob_att_test, ycn_pred_att_test,  ycn_metric_att_test   = y_prediction(  baseline_model_att_cn, xcn_test_np, xcn_test, y_test,   "ycn_pred_att")
    ykc_prob_att_test, ykc_pred_att_test,  ykc_metric_att_test   = y_prediction(  baseline_model_att_kc, xkc_test_np, xkc_test, y_test,   "ykc_pred_att")
    yce_prob_att_test, yce_pred_att_test,  yce_metric_att_test   = y_prediction(  baseline_model_att_ce, xce_test_np, xce_test, y_test,   "yce_pred_att")
    ysc_prob_att_test, ysc_pred_att_test,  ysc_metric_att_test   = y_prediction(  baseline_model_att_sc, xsc_test_np, xsc_test, y_test,   "ysc_pred_att")
    yac_prob_att_test, yac_pred_att_test,  yac_metric_att_test   = y_prediction(  baseline_model_att_ac, xac_test_np, xac_test, y_test,   "yac_pred_att")
    yma_prob_att_test, yma_pred_att_test,  yma_metric_att_test   = y_prediction(  baseline_model_att_ma, xma_test_np, xma_test, y_test,   "yma_pred_att")
    
    # Stack the predictive features
    stack_train_prob_all = pd.concat([yat_prob_cnn_train, yat_prob_bilstm_train, yat_prob_att_train,
                            yes_prob_cnn_train, yes_prob_bilstm_train, yes_prob_att_train,
                            yke_prob_cnn_train, yke_prob_bilstm_train, yke_prob_att_train,
                            ypc_prob_cnn_train, ypc_prob_bilstm_train, ypc_prob_att_train,
                            yss_prob_cnn_train, yss_prob_bilstm_train, yss_prob_att_train,
                            ycd_prob_cnn_train, ycd_prob_bilstm_train, ycd_prob_att_train,
                            ycn_prob_cnn_train, ycn_prob_bilstm_train, ycn_prob_att_train,
                            ykc_prob_cnn_train, ykc_prob_bilstm_train, ykc_prob_att_train,
                            yce_prob_cnn_train, yce_prob_bilstm_train, yce_prob_att_train,
                            ysc_prob_cnn_train, ysc_prob_bilstm_train, ysc_prob_att_train,
                            yac_prob_cnn_train, yac_prob_bilstm_train, yac_prob_att_train,
                            yma_prob_cnn_train, yma_prob_bilstm_train, yma_prob_att_train],  axis=1)
    stack_test_prob_all  = pd.concat([yat_prob_cnn_test, yat_prob_bilstm_test, yat_prob_att_test,
                            yes_prob_cnn_test, yes_prob_bilstm_test, yes_prob_att_test,
                            yke_prob_cnn_test, yke_prob_bilstm_test, yke_prob_att_test,
                            ypc_prob_cnn_test, ypc_prob_bilstm_test, ypc_prob_att_test,
                            yss_prob_cnn_test, yss_prob_bilstm_test, yss_prob_att_test,
                            ycd_prob_cnn_test, ycd_prob_bilstm_test, ycd_prob_att_test,
                            ycn_prob_cnn_test, ycn_prob_bilstm_test, ycn_prob_att_test,
                            ykc_prob_cnn_test, ykc_prob_bilstm_test, ykc_prob_att_test,
                            yce_prob_cnn_test, yce_prob_bilstm_test, yce_prob_att_test,
                            ysc_prob_cnn_test, ysc_prob_bilstm_test, ysc_prob_att_test,
                            yac_prob_cnn_test, yac_prob_bilstm_test, yac_prob_att_test,
                            yma_prob_cnn_test, yma_prob_bilstm_test, yma_prob_att_test],  axis=1)
    stack_test_pred_all  = pd.concat([yat_pred_cnn_test, yat_pred_bilstm_test, yat_pred_att_test,
                        yes_pred_cnn_test, yes_pred_bilstm_test, yes_pred_att_test,
                        yke_pred_cnn_test, yke_pred_bilstm_test, yke_pred_att_test,
                        ypc_pred_cnn_test, ypc_pred_bilstm_test, ypc_pred_att_test,
                        yss_pred_cnn_test, yss_pred_bilstm_test, yss_pred_att_test,
                        ycd_pred_cnn_test, ycd_pred_bilstm_test, ycd_pred_att_test,
                        ycn_pred_cnn_test, ycn_pred_bilstm_test, ycn_pred_att_test,
                        ykc_pred_cnn_test, ykc_pred_bilstm_test, ykc_pred_att_test,
                        yce_pred_cnn_test, yce_pred_bilstm_test, yce_pred_att_test,
                        ysc_pred_cnn_test, ysc_pred_bilstm_test, ysc_pred_att_test,
                        yac_pred_cnn_test, yac_pred_bilstm_test, yac_pred_att_test,
                        yma_pred_cnn_test, yma_pred_bilstm_test, yma_pred_att_test],  axis=1)
    stack_train_pred_all  = pd.concat([yat_pred_cnn_train, yat_pred_bilstm_train, yat_pred_att_train,
                        yes_pred_cnn_train, yes_pred_bilstm_train, yes_pred_att_train,
                        yke_pred_cnn_train, yke_pred_bilstm_train, yke_pred_att_train,
                        ypc_pred_cnn_train, ypc_pred_bilstm_train, ypc_pred_att_train,
                        yss_pred_cnn_train, yss_pred_bilstm_train, yss_pred_att_train,
                        ycd_pred_cnn_train, ycd_pred_bilstm_train, ycd_pred_att_train,
                        ycn_pred_cnn_train, ycn_pred_bilstm_train, ycn_pred_att_train,
                        ykc_pred_cnn_train, ykc_pred_bilstm_train, ykc_pred_att_train,
                        yce_pred_cnn_train, yce_pred_bilstm_train, yce_pred_att_train,
                        ysc_pred_cnn_train, ysc_pred_bilstm_train, ysc_pred_att_train,
                        yac_pred_cnn_train, yac_pred_bilstm_train, yac_pred_att_train,
                        yma_pred_cnn_train, yma_pred_bilstm_train, yma_pred_att_train],  axis=1)

    stack_train_prob_all.to_csv (os.path.join(name, "all_stacked_train_prob.csv"))
    stack_test_prob_all.to_csv  (os.path.join(name, "all_stacked_test_prob.csv"))
    stack_train_pred_all.to_csv (os.path.join(name, "all_stacked_train_predict.csv"))
    stack_test_pred_all.to_csv  (os.path.join(name, "all_stacked_test_predict.csv"))

    # Combine performance metrics
    metric_train= pd.concat([yat_metric_cnn_train, yat_metric_bilstm_train, yat_metric_att_train,
                            yes_metric_cnn_train, yes_metric_bilstm_train, yes_metric_att_train,
                            yke_metric_cnn_train, yke_metric_bilstm_train, yke_metric_att_train,
                            ypc_metric_cnn_train, ypc_metric_bilstm_train, ypc_metric_att_train,
                            yss_metric_cnn_train, yss_metric_bilstm_train, yss_metric_att_train,
                            ycd_metric_cnn_train, ycd_metric_bilstm_train, ycd_metric_att_train,
                            ycn_metric_cnn_train, ycn_metric_bilstm_train, ycn_metric_att_train,
                            ykc_metric_cnn_train, ykc_metric_bilstm_train, ykc_metric_att_train,
                            yce_metric_cnn_train, yce_metric_bilstm_train, yce_metric_att_train,
                            ysc_metric_cnn_train, ysc_metric_bilstm_train, ysc_metric_att_train,
                            yac_metric_cnn_train, yac_metric_bilstm_train, yac_metric_att_train,
                            yma_metric_cnn_train, yma_metric_bilstm_train, yma_metric_att_train],  axis=0)
    metric_test= pd.concat([yat_metric_cnn_test, yat_metric_bilstm_test, yat_metric_att_test,
                            yes_metric_cnn_test, yes_metric_bilstm_test, yes_metric_att_test,
                            yke_metric_cnn_test, yke_metric_bilstm_test, yke_metric_att_test,
                            ypc_metric_cnn_test, ypc_metric_bilstm_test, ypc_metric_att_test,
                            yss_metric_cnn_test, yss_metric_bilstm_test, yss_metric_att_test,
                            ycd_metric_cnn_test, ycd_metric_bilstm_test, ycd_metric_att_test,
                            ycn_metric_cnn_test, ycn_metric_bilstm_test, ycn_metric_att_test,
                            ykc_metric_cnn_test, ykc_metric_bilstm_test, ykc_metric_att_test,
                            yce_metric_cnn_test, yce_metric_bilstm_test, yce_metric_att_test,
                            ysc_metric_cnn_test, ysc_metric_bilstm_test, ysc_metric_att_test,
                            yac_metric_cnn_test, yac_metric_bilstm_test, yac_metric_att_test,
                            yma_metric_cnn_test, yma_metric_bilstm_test, yma_metric_att_test],  axis=0)
    metric_train.to_csv(os.path.join( name, "metric_train.csv"))
    metric_test.to_csv(os.path.join( name, "metric_test.csv"))

def main():
    for name in ['margin1/train_1']: # Change according to running file
        print("#"*100) 
        print(name)
        y_train  = pd.read_csv(os.path.join(name, "y_train.csv"), index_col=0)
        print(y_train)
        stacked_class(name)
        print("Finish training model ", name)

if __name__ == "__main__":
    main() 