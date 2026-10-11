import numpy as np
from fetch import Get2dMapTrainFold, Get2dMapTestData

def load_level_data(level, val_fold=5):
    """
    Loads all 5 folds for a given level using fetch.py.
    Combines 4 folds for training and keeps 1 for validation.
    """
    X_train_list = []
    Y_train_list = []
    
    X_val = None
    Y_val = None
    
    print(f"Loading data for Level {level} (Validation on Fold {val_fold})...")
    
    for fold in range(1, 6):
        # 1. Fetch data from memory map
        X, Y = Get2dMapTrainFold(level, fold)
        
        # 2. Convert to standard numpy arrays (loads into RAM)
        X = np.array(X, dtype=np.float32) 
        Y = np.array(Y, dtype=np.float32)
        
        X = np.log10(np.maximum(X, 1e-5))
        # 3. Add channel dimension for PyTorch: (Batch, Channel, Height, Width)
        # Note: If you are using TensorFlow/Keras, change axis=1 to axis=-1
        X = np.expand_dims(X, axis=1) 
        
        # 4. Route the data to either Validation or Training
        if fold == val_fold:
            X_val, Y_val = X, Y
        else:
            X_train_list.append(X)
            Y_train_list.append(Y)
            
    # 5. Concatenate the 4 training folds into single arrays
    X_train = np.concatenate(X_train_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)

    mean = np.mean(X_train)
    std = np.std(X_train)
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    
    return X_train, Y_train, X_val, Y_val


# ==========================================
# TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # Test Level 1
    level_to_test = 1
    
    X_train, Y_train, X_val, Y_val = load_level_data(level=level_to_test, val_fold=5)
    
    print("\n--- SUCCESS! Data shapes ---")
    print(f"X_train shape (4 folds):  {X_train.shape} -> Expected ~ (660, 1, 256, 256)")
    print(f"Y_train shape (4 folds):  {Y_train.shape} -> Expected ~ (660, 2)")
    print(f"X_val shape   (1 fold):   {X_val.shape} -> Expected ~ (165, 1, 256, 256)")
    print(f"Y_val shape   (1 fold):   {Y_val.shape} -> Expected ~ (165, 2)")
    
    # Optional: Quick check of the Final Test Data just to be sure
    print("\nLoading Final Test Data...")
    X_test, Y_test = Get2dMapTestData(level=level_to_test)
    X_test = np.expand_dims(np.array(X_test, dtype=np.float32), axis=1)
    Y_test = np.array(Y_test, dtype=np.float32)
    print(f"X_test shape: {X_test.shape}")