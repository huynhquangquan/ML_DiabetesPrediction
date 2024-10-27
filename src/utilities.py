import pickle
from pathlib import Path
import yaml
import pandas as pd
import joblib
import yaml
import importlib
from sklearn.preprocessing import StandardScaler

# Read YAML
def imputation_select():
    dir = Path(__file__).parent.parent
    with open(dir / 'config' / 'preprocess_config.yaml','r') as file:
        imputation = yaml.safe_load(file)
    return imputation['imputation']

def dataset_select(): # change dataset in config.dataset_config
    dir = Path(__file__).parent.parent
    with open(dir / 'config' / 'dataset_config.yaml','r') as file:
        dataset = yaml.safe_load(file)
    return dataset

def model_select(): # every run in evaluate, prediction, visualize, will select the model to run, change model in config.model_select
    dir = Path(__file__).parent.parent
    with open(dir / 'config' / 'model_select.yaml','r') as file:
        model = yaml.safe_load(file)
    return model['model']

def scaling_config():
    dir = Path(__file__).parent.parent
    with open(dir / 'config' / 'preprocess_config.yaml','r') as file:
        scale = yaml.safe_load(file)
    return scale['scaling']

def balance_config():
    dir = Path(__file__).parent.parent
    with open(dir / 'config' / 'preprocess_config.yaml','r') as file:
        balance = yaml.safe_load(file)
    return balance['balance_method']

def outliers_config():
    dir = Path(__file__).parent.parent
    with open(dir / 'config' / 'preprocess_config.yaml','r') as file:
        outliers = yaml.safe_load(file)
    return outliers['outliers_method']

# Function
def dynamic_import_imputation(imputation_method):
    try:
        # Dynamically import the imputation method from the src.preprocessing.imputation module
        module = importlib.import_module(f'src.preprocessing.imputation.{imputation_method}')
        impute_function = getattr(module, 'replace_missing_values')
        return impute_function
    except ModuleNotFoundError as e:
        raise ImportError(f"Module not found for {imputation_method}: {e}")
    except AttributeError as e:
        raise ImportError(f"Function replace_missing_values not found for {imputation_method}: {e}")

def dynamic_import_balance(balance_method):
    try:
        # Dynamically import the imputation method from the src.preprocessing.imputation module
        module = importlib.import_module(f'src.preprocessing.balance.{balance_method}')
        impute_function = getattr(module, 'balance_train')
        return impute_function
    except ModuleNotFoundError as e:
        raise ImportError(f"Module not found for {balance_method}: {e}")
    except AttributeError as e:
        raise ImportError(f"Function balance_train not found for {balance_method}: {e}")

def dynamic_import_outliers(outliers_method):
    try:
        # Dynamically import the imputation method from the src.preprocessing.imputation module
        module = importlib.import_module(f'src.preprocessing.outliers.{outliers_method}')
        impute_function = getattr(module, 'remove_outliers')
        return impute_function
    except ModuleNotFoundError as e:
        raise ImportError(f"Module not found for {outliers_method}: {e}")
    except AttributeError as e:
        raise ImportError(f"Function remove_outliers not found for {outliers_method}: {e}")

def find_path_from_models(name): # Side function to create path in .virtualenvironment.models
    dir = Path(__file__).parent
    model_path = dir / '..' / "models" / f'{name}'
    return model_path.resolve()

def check_model(name):
    try:
        check_models = joblib_load(name)
    except:
        print(f'Không có mô hình {name}')
        return False
    return True

def check_raw():
    try:
        check_raw = read_raw("diabetes.csv")
    except:
        print(f'Chưa có dataset')
        return False
    return True

def check_processed():
    try:
        check_test = read_processed("test.csv")
        check_train = read_processed("train.csv")
    except:
        print(f'Processed không có hoặc không đầy đủ')
        return False
    return True

def read_raw(datasetname):
    current_dir = Path(__file__).parent
    model_path = current_dir / '..' / 'data' / 'raw' / f'{datasetname}'
    df = pd.read_csv(model_path.resolve())
    return df

def read_processed(datasetname):
    current_dir = Path(__file__).parent
    model_path = current_dir / '..' / 'data' / 'processed' / f'{datasetname}'
    df = pd.read_csv(model_path.resolve())
    return df

def scaling(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def joblib_dump(model,name):
    model_path = find_path_from_models(name)
    joblib.dump(model,model_path)

def joblib_load(name):
    model_path = find_path_from_models(name)
    loaded = joblib.load(model_path)
    return loaded
