import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score,recall_score, matthews_corrcoef, f1_score, precision_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Define the evaluation function
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

# Wrapper class for Keras model to provide predict_proba method
class KerasClassifierWrapper:
    def __init__(self, keras_model):
        self.model = keras_model

    def predict_proba(self, X):
        return self.model.predict(X)

def activelearning(name, fingerprint, classifier):
    # Wrap the classifier
    wrapped_classifier = KerasClassifierWrapper(classifier)
    
    x_subset = pd.read_csv(os.path.join(name, "train", f'{fingerprint}.csv'), index_col=0)
    y_subset = pd.read_csv(os.path.join(name, "train", "y_subset.csv"  ), index_col=0)
    x_pool   = pd.read_csv(os.path.join(name, "pool", f'{fingerprint}.csv'), index_col=0)
    y_pool   = pd.read_csv(os.path.join(name, "pool", "y_pool.csv"  ), index_col=0)
    x_test   = pd.read_csv(os.path.join(name, "test", f'{fingerprint}.csv'), index_col=0)
    y_test   = pd.read_csv(os.path.join(name, "test", "y_test.csv"  ), index_col=0)
    
    x_subset_np = np.array(x_subset)
    y_subset_np = np.array(y_subset)
    x_pool_np   = np.array(x_pool)
    y_pool_np   = np.array(y_pool)
    x_test_np   = np.array(x_test)
    y_test_np   = np.array(y_test)
    print("x_pool columns:", x_pool.columns)
    
    print("Evaluating on pool data:")
    pool_y_prob, pool_y_pred, pool_metrics = y_prediction(classifier, x_pool_np, x_pool,  y_pool_np, col_name="y_pool")
    pool_y_prob.to_csv(os.path.join (name, "iteration_pool_y_prob.csv"), index=True)
    pool_y_pred.to_csv(os.path.join (name, "iteration_pool_y_pred.csv"), index=True)
    pool_metrics.to_csv(os.path.join(name, "iteration_pool_metrics.csv"), index=True)
    
    x_pool.reset_index(drop=True, inplace=True)
    y_pool.reset_index(drop=True, inplace=True)
    #   5% of data set
    n_instances = int(0.05 * len(x_pool))
    n_instances = max(1, n_instances)
    
    #   Query the next batch of data
    selected_indices, selected_samples = margin_sampling(   # define strategies
        classifier=wrapped_classifier,
        X=x_pool,
        n_instances=n_instances,
        random_tie_break=True
    )
    
    x_selected = x_pool.iloc[selected_indices]
    y_selected = y_pool.iloc[selected_indices]
    print("Selected rows from x_pool:", x_selected)
    print("Selected compounds from x_pool:", x_selected.shape)
    print("Selected indices:", selected_indices)
    
    x_subset = np.concatenate([x_subset, x_selected.values])
    y_subset = np.concatenate([y_subset, y_selected.values])
    print("Total of x_subset and y_subset:", x_subset.shape, y_subset.shape)
    
    x_subset_df = pd.DataFrame(x_subset, columns=x_pool.columns)
    y_subset_df = pd.DataFrame(y_subset, columns=y_pool.columns)
    x_subset_df.to_csv(os.path.join(name, "iteration_1_x_subset.csv"), index=False)
    y_subset_df.to_csv(os.path.join(name, "iteration_1_y_subset.csv"), index=False)
    
    x_pool = x_pool.drop(selected_indices).reset_index(drop=True)
    y_pool = y_pool.drop(selected_indices).reset_index(drop=True)
    x_pool.to_csv(os.path.join(name, "iteration_1_x_pool.csv"), index=False)
    y_pool.to_csv(os.path.join(name, "iteration_1_y_pool.csv"), index=False)
    print("Total of x_pool and y_pool:", x_pool.shape, y_pool.shape)
    
    x_subset_np = np.array(x_subset)
    y_subset_np = np.array(y_subset)
    x_pool_np = np.array(x_pool)
    y_pool_np = np.array(y_pool)
    
    x_train_stack, x_val_stack, y_train_stack, y_val_stack = train_test_split(x_subset_np, y_subset_np, test_size=0.2, random_state=42)
    classifier.fit(x_train_stack,  y_train_stack, validation_data=(x_val_stack, y_val_stack), epochs=20, batch_size=32)  # Retraining the model with the updated dataset
    classifier.save(os.path.join(name, "iteration_1_model.keras"))
    
    # Evaluate performance on the test and pool data
    print("Evaluating on test data:")
    classifier = load_model(os.path.join(name, f"iteration_1_model.keras"))
    test_y_prob, test_y_pred, test_metrics = y_prediction(classifier, x_test_np, x_test, y_test_np, col_name=f"1_y_test_pred")
    test_y_prob.to_csv (os.path.join(name, "iteration_1_test_y_prob.csv"), index=True)
    test_y_pred.to_csv (os.path.join(name, "iteration_1_test_y_pred.csv"), index=True)
    test_metrics.to_csv(os.path.join(name, "iteration_1_test_metrics.csv"), index=True)
    
    print("Evaluating on pool data:")
    pool_y_prob, pool_y_pred, pool_metrics = y_prediction(classifier, x_pool_np, x_pool,  y_pool_np, col_name=f"1_y_pool_pred")
    pool_y_prob.to_csv (os.path.join(name, "iteration_1_pool_y_prob.csv"), index=True)
    pool_y_pred.to_csv (os.path.join(name, "iteration_1_pool_y_pred.csv"), index=True)
    pool_metrics.to_csv(os.path.join(name, "iteration_1_pool_metrics.csv"), index=True)
    
    # Active Learning Loop for the rest iterations
    for i in range(2, 5):
        print(f"Iteration {i}")
        
        classifier = load_model(os.path.join(name, f"iteration_{i-1}_model.keras"))
        wrapped_classifier = KerasClassifierWrapper(classifier)

        x_pool   = pd.read_csv(os.path.join(name, f"iteration_{i-1}_x_pool.csv"))
        y_pool   = pd.read_csv(os.path.join(name, f"iteration_{i-1}_y_pool.csv"))
        
        x_pool.reset_index(drop=True, inplace=True)
        y_pool.reset_index(drop=True, inplace=True)
        #   5% of data set
        n_instances = int(0.05 * len(x_pool))
        n_instances = max(1, n_instances)

        #   Query the next batch of data
        selected_indices, selected_samples = margin_sampling(   # define strategies
            classifier=wrapped_classifier,
            X=x_pool,
            n_instances=n_instances,
            random_tie_break=True
        )
        
        x_selected = x_pool.iloc[selected_indices]
        y_selected = y_pool.iloc[selected_indices]
        print("Selected rows from x_pool:", x_selected)
        print("Selected compounds from x_pool:", x_selected.shape)
        print("Selected indices:", selected_indices)

        x_subset = np.concatenate([x_subset, x_selected.values])
        y_subset = np.concatenate([y_subset, y_selected.values])
        print("Total of x_subset and y_subset:", x_subset.shape, y_subset.shape)
        
        x_subset_df = pd.DataFrame(x_subset, columns=x_pool.columns)
        y_subset_df = pd.DataFrame(y_subset, columns=y_pool.columns)
        x_subset_df.to_csv(os.path.join(name, f"iteration_{i}_x_subset.csv"), index=False)
        y_subset_df.to_csv(os.path.join(name, f"iteration_{i}_y_subset.csv"), index=False)
        
        x_pool = x_pool.drop(selected_indices).reset_index(drop=True)
        y_pool = y_pool.drop(selected_indices).reset_index(drop=True)
        x_pool.to_csv(os.path.join(name, f"iteration_{i}_x_pool.csv"), index=False)
        y_pool.to_csv(os.path.join(name, f"iteration_{i}_y_pool.csv"), index=False)
        print("Total of x_pool and y_pool:", x_pool.shape, y_pool.shape)
        
        x_subset_np = np.array(x_subset)
        y_subset_np = np.array(y_subset)
        x_pool_np = np.array(x_pool)
        y_pool_np = np.array(y_pool)

        x_train_stack, x_val_stack, y_train_stack, y_val_stack = train_test_split(x_subset_np, y_subset_np, test_size=0.2, random_state=42)
        classifier.fit(x_train_stack,  y_train_stack, validation_data=(x_val_stack, y_val_stack), epochs=20, batch_size=32)  # Retraining the model with the updated dataset
        classifier.save(os.path.join(name, f"iteration_{i}_model.keras"))
        
        # Evaluate performance on the test and pool data
        print("Evaluating on test data:")
        classifier = load_model(os.path.join(name, f"iteration_{i}_model.keras"))
        test_y_prob, test_y_pred, test_metrics = y_prediction(classifier, x_test_np, x_test, y_test_np, col_name=f"{i}_y_test_pred")
        test_y_prob.to_csv (os.path.join(name, f"iteration_{i}_test_y_prob.csv"), index=True)
        test_y_pred.to_csv (os.path.join(name, f"iteration_{i}_test_y_pred.csv"), index=True)
        test_metrics.to_csv(os.path.join(name, f"iteration_{i}_test_metrics.csv"), index=True)

        print("Evaluating on pool data:")
        pool_y_prob, pool_y_pred, pool_metrics = y_prediction(classifier, x_pool_np, x_pool,  y_pool_np, col_name=f"{i}_y_pool")
        pool_y_prob.to_csv (os.path.join(name, f"iteration_{i}_pool_y_prob.csv"), index=True)
        pool_y_pred.to_csv (os.path.join(name, f"iteration_{i}_pool_y_pred.csv"), index=True)
        pool_metrics.to_csv(os.path.join(name, f"iteration_{i}_pool_metrics.csv"), index=True)
        
    print("Active learning completed.")

def main():
    name = 'run_al_0.1'
    fingerprint = 'KRFP'
    classifier = load_model("run_subset_0.1/baseline_model_att_ke.keras")
    activelearning(name, fingerprint, classifier)
    
if __name__ == "__main__":
    main()
