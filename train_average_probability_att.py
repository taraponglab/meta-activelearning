import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Attention, Dense, Reshape
import pandas as pd
import numpy as np
import os
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, precision_score
from sklearn.model_selection import train_test_split

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

def meta_cnn_train(iteration_folder, name, test_folder='data/test'):
    '''
    This function is to train the meta-model based attention
    '''
    
    x1_train = pd.read_csv(os.path.join( iteration_folder, 'train_1', "all_stacked_train_oof.csv"  ), index_col=0)
    x2_train = pd.read_csv(os.path.join( iteration_folder, 'train_2', "all_stacked_train_oof.csv"  ), index_col=0)
    x3_train = pd.read_csv(os.path.join( iteration_folder, 'train_3', "all_stacked_train_oof.csv"  ), index_col=0)
    x4_train = pd.read_csv(os.path.join( iteration_folder, 'train_4', "all_stacked_train_oof.csv"  ), index_col=0)
    x5_train = pd.read_csv(os.path.join( iteration_folder, 'train_5', "all_stacked_train_oof.csv"  ), index_col=0)
    x6_train = pd.read_csv(os.path.join( iteration_folder, 'train_6', "all_stacked_train_oof.csv"  ), index_col=0)
    
    y1_train = pd.read_csv(os.path.join( iteration_folder, 'train_1',   "y_train.csv"  ), index_col=0)
    y2_train = pd.read_csv(os.path.join( iteration_folder, 'train_2',   "y_train.csv"  ), index_col=0)
    y3_train = pd.read_csv(os.path.join( iteration_folder, 'train_3',   "y_train.csv"  ), index_col=0)
    y4_train = pd.read_csv(os.path.join( iteration_folder, 'train_4',   "y_train.csv"  ), index_col=0)
    y5_train = pd.read_csv(os.path.join( iteration_folder, 'train_5',   "y_train.csv"  ), index_col=0)
    y6_train = pd.read_csv(os.path.join( iteration_folder, 'train_6',   "y_train.csv"  ), index_col=0)
    
    x1_test = pd.read_csv(os.path.join( iteration_folder, 'train_1', "all_stacked_test_prob.csv"  ), index_col=0)
    x2_test = pd.read_csv(os.path.join( iteration_folder, 'train_2', "all_stacked_test_prob.csv"  ), index_col=0)
    x3_test = pd.read_csv(os.path.join( iteration_folder, 'train_3', "all_stacked_test_prob.csv"  ), index_col=0)
    x4_test = pd.read_csv(os.path.join( iteration_folder, 'train_4', "all_stacked_test_prob.csv"  ), index_col=0)
    x5_test = pd.read_csv(os.path.join( iteration_folder, 'train_5', "all_stacked_test_prob.csv"  ), index_col=0)
    x6_test = pd.read_csv(os.path.join( iteration_folder, 'train_6', "all_stacked_test_prob.csv"  ), index_col=0)
    y_test = pd.read_csv(os.path.join( test_folder, "y_test.csv"  ), index_col=0)
    
    x1_train_np = np.array(x1_train)
    x2_train_np = np.array(x2_train)
    x3_train_np = np.array(x3_train)
    x4_train_np = np.array(x4_train)
    x5_train_np = np.array(x5_train)
    x6_train_np = np.array(x6_train)
    
    y1_train = np.array(y1_train)
    y2_train = np.array(y2_train)
    y3_train = np.array(y3_train)
    y4_train = np.array(y4_train)
    y5_train = np.array(y5_train)
    y6_train = np.array(y6_train)
    
    x1_test_np = np.array(x1_test)
    x2_test_np = np.array(x2_test)
    x3_test_np = np.array(x3_test)
    x4_test_np = np.array(x4_test)
    x5_test_np = np.array(x5_test)
    x6_test_np = np.array(x6_test)
    y_test = np.array(y_test)
    
    x1_train_split, x1_val_split, y1_train_split, y1_val_split = train_test_split(x1_train_np, y1_train, test_size=0.2, random_state=42)
    x2_train_split, x2_val_split, y2_train_split, y2_val_split = train_test_split(x2_train_np, y2_train, test_size=0.2, random_state=42)
    x3_train_split, x3_val_split, y3_train_split, y3_val_split = train_test_split(x3_train_np, y3_train, test_size=0.2, random_state=42)
    x4_train_split, x4_val_split, y4_train_split, y4_val_split = train_test_split(x4_train_np, y4_train, test_size=0.2, random_state=42)
    x5_train_split, x5_val_split, y5_train_split, y5_val_split = train_test_split(x5_train_np, y5_train, test_size=0.2, random_state=42)
    x6_train_split, x6_val_split, y6_train_split, y6_val_split = train_test_split(x6_train_np, y6_train, test_size=0.2, random_state=42)
    
    meta_model_attention_1 = attention(fingerprint_length=x1_train_split.shape[1])
    meta_model_attention_2 = attention(fingerprint_length=x2_train_split.shape[1])
    meta_model_attention_3 = attention(fingerprint_length=x3_train_split.shape[1])
    meta_model_attention_4 = attention(fingerprint_length=x4_train_split.shape[1])
    meta_model_attention_5 = attention(fingerprint_length=x5_train_split.shape[1])
    meta_model_attention_6 = attention(fingerprint_length=x6_train_split.shape[1])
    
    meta_model_attention_1.fit(x1_train_split, y1_train_split, validation_data=(x1_val_split, y1_val_split), epochs=20, batch_size=32)
    meta_model_attention_2.fit(x2_train_split, y2_train_split, validation_data=(x2_val_split, y2_val_split), epochs=20, batch_size=32)
    meta_model_attention_3.fit(x3_train_split, y3_train_split, validation_data=(x3_val_split, y3_val_split), epochs=20, batch_size=32)
    meta_model_attention_4.fit(x4_train_split, y4_train_split, validation_data=(x4_val_split, y4_val_split), epochs=20, batch_size=32)
    meta_model_attention_5.fit(x5_train_split, y5_train_split, validation_data=(x5_val_split, y5_val_split), epochs=20, batch_size=32)
    meta_model_attention_6.fit(x6_train_split, y6_train_split, validation_data=(x6_val_split, y6_val_split), epochs=20, batch_size=32)
    
    meta_model_attention_1.save(os.path.join(name, "meta_model_attention_1.keras"))
    meta_model_attention_2.save(os.path.join(name, "meta_model_attention_2.keras"))
    meta_model_attention_3.save(os.path.join(name, "meta_model_attention_3.keras"))
    meta_model_attention_4.save(os.path.join(name, "meta_model_attention_4.keras"))
    meta_model_attention_5.save(os.path.join(name, "meta_model_attention_5.keras"))
    meta_model_attention_6.save(os.path.join(name, "meta_model_attention_6.keras"))
    
    y1_prob_attention_train, y1_pred_attention_train, y1_metric_attention_train = y_prediction(meta_model_attention_1, x1_train_np, x1_train, y1_train, "y1_pred_attention")
    y2_prob_attention_train, y2_pred_attention_train, y2_metric_attention_train = y_prediction(meta_model_attention_2, x2_train_np, x2_train, y2_train, "y2_pred_attention")
    y3_prob_attention_train, y3_pred_attention_train, y3_metric_attention_train = y_prediction(meta_model_attention_3, x3_train_np, x3_train, y3_train, "y3_pred_attention")
    y4_prob_attention_train, y4_pred_attention_train, y4_metric_attention_train = y_prediction(meta_model_attention_4, x4_train_np, x4_train, y4_train, "y4_pred_attention")
    y5_prob_attention_train, y5_pred_attention_train, y5_metric_attention_train = y_prediction(meta_model_attention_5, x5_train_np, x5_train, y5_train, "y5_pred_attention")
    y6_prob_attention_train, y6_pred_attention_train, y6_metric_attention_train = y_prediction(meta_model_attention_6, x6_train_np, x6_train, y6_train, "y6_pred_attention")
    y1_prob_attention_test, y1_pred_attention_test, y1_metric_attention_test   =  y_prediction(meta_model_attention_1, x1_test_np, x1_test, y_test, "y1_pred_attention")
    y2_prob_attention_test, y2_pred_attention_test, y2_metric_attention_test   =  y_prediction(meta_model_attention_2, x2_test_np, x2_test, y_test, "y2_pred_attention")
    y3_prob_attention_test, y3_pred_attention_test, y3_metric_attention_test   =  y_prediction(meta_model_attention_3, x3_test_np, x3_test, y_test, "y3_pred_attention")
    y4_prob_attention_test, y4_pred_attention_test, y4_metric_attention_test   =  y_prediction(meta_model_attention_4, x4_test_np, x4_test, y_test, "y4_pred_attention")
    y5_prob_attention_test, y5_pred_attention_test, y5_metric_attention_test   =  y_prediction(meta_model_attention_5, x5_test_np, x5_test, y_test, "y5_pred_attention")
    y6_prob_attention_test, y6_pred_attention_test, y6_metric_attention_test   =  y_prediction(meta_model_attention_6, x6_test_np, x6_test, y_test, "y6_pred_attention")
    
    y1_prob_attention_train.to_csv(os.path.join( name, "y1_prob_train.csv"))
    y2_prob_attention_train.to_csv(os.path.join( name, "y2_prob_train.csv"))
    y3_prob_attention_train.to_csv(os.path.join( name, "y3_prob_train.csv"))
    y4_prob_attention_train.to_csv(os.path.join( name, "y4_prob_train.csv"))
    y5_prob_attention_train.to_csv(os.path.join( name, "y5_prob_train.csv"))
    y6_prob_attention_train.to_csv(os.path.join( name, "y6_prob_train.csv"))
    y1_pred_attention_train.to_csv(os.path.join( name, "y1_pred_train.csv"))
    y2_pred_attention_train.to_csv(os.path.join( name, "y2_pred_train.csv"))
    y3_pred_attention_train.to_csv(os.path.join( name, "y3_pred_train.csv"))
    y4_pred_attention_train.to_csv(os.path.join( name, "y4_pred_train.csv"))
    y5_pred_attention_train.to_csv(os.path.join( name, "y5_pred_train.csv"))
    y6_pred_attention_train.to_csv(os.path.join( name, "y6_pred_train.csv"))
    
    stack_test_prob = pd.concat([y1_prob_attention_test,
                        y2_prob_attention_test,
                        y3_prob_attention_test,
                        y4_prob_attention_test,
                        y5_prob_attention_test,
                        y6_prob_attention_test],  axis=1)
    stack_test_pred = pd.concat([y1_pred_attention_test,
                    y2_pred_attention_test,
                    y3_pred_attention_test,
                    y4_pred_attention_test,
                    y5_pred_attention_test,
                    y6_pred_attention_test],  axis=1)
    stack_test_prob.to_csv(os.path.join( name, "stack_test_prob.csv"))
    stack_test_pred.to_csv(os.path.join( name, "stack_test_pred.csv"))
    
    stack_test_prob = pd.read_csv(os.path.join( name, "stack_test_prob.csv"), index_col=0)
    stack_test_prob["y_prob_test_average"] = stack_test_prob.mean(axis=1)
    output_file = os.path.join(name, "y_prob_test_average.csv")
    stack_test_prob[["y_prob_test_average"]].to_csv(output_file)
    x_test_prob = pd.read_csv(os.path.join( name, "y_prob_test_average.csv"), index_col=0)
    y_pred_test_average, metrics_test_average = y_prediction_average(x_test_prob, y_test, "y_pred_test_average")
    y_pred_test_average.to_csv(os.path.join( name, "y_pred_test_average.csv"))
    
    metric_train= pd.concat([y1_metric_attention_train,
                            y2_metric_attention_train,
                            y3_metric_attention_train,
                            y4_metric_attention_train,
                            y5_metric_attention_train,
                            y6_metric_attention_train],  axis=0)
    metric_test= pd.concat([y1_metric_attention_test,
                        y2_metric_attention_test,
                        y3_metric_attention_test,
                        y4_metric_attention_test,
                        y5_metric_attention_test,
                        y6_metric_attention_test, metrics_test_average],  axis=0)
    
    metric_train.to_csv(os.path.join( name, "metric_train.csv"))
    metric_test.to_csv(os.path.join( name, "metric_test.csv"))

def main():
    '''
    This function is to train the meta-model based CNN
    It prompts the user to input the iteration folder and subfolder names,
    creates the necessary directories, and trains the CNN model for each subfolder.
    '''
    iteration_folder = input("Enter the iteration folder (e.g., subset1): ").strip()
    subfolders = 'meta_average_att' #input("Enter subfolder names separated by commas (e.g., meta_average_att): ")
    subfolder_list = [sf.strip() for sf in subfolders.split(",") if sf.strip()]
    for subfolder in subfolder_list:
        name = os.path.join(iteration_folder, subfolder)
        os.makedirs(name, exist_ok=True)  # Automatically create the folder if it doesn't exist
        print("#" * 100)
        print(name)
        meta_cnn_train(iteration_folder, name, test_folder='data/test')
        print("finish train ", name)
        
if __name__ == "__main__":
    main()
