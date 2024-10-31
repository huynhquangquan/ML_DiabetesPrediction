import pickle
from pathlib import Path
import yaml
import pandas as pd
import joblib
import yaml
import importlib
from sklearn.preprocessing import RobustScaler
from src.preprocessing import scaler

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

def scaler_select():
    dir = Path(__file__).parent.parent
    with open(dir / 'config' / 'preprocess_config.yaml','r') as file:
        scaler = yaml.safe_load(file)
    return scaler['scaler']

def balance_select():
    dir = Path(__file__).parent.parent
    with open(dir / 'config' / 'preprocess_config.yaml','r') as file:
        balance = yaml.safe_load(file)
    return balance['balance_method']

def outliers_select():
    dir = Path(__file__).parent.parent
    with open(dir / 'config' / 'preprocess_config.yaml','r') as file:
        outliers = yaml.safe_load(file)
    return outliers['outliers_method']

def model_ensemble():
    dir = Path(__file__).parent.parent
    with open(dir / 'config' / 'model_select.yaml','r') as file:
        model = yaml.safe_load(file)
    return model['model_ensemble']['model1'], model['model_ensemble']['model2'], model['model_ensemble']['model3']

# Function
def dynamic_import_scaler(scaler_method):
    try:
        # Dynamically import the imputation method from the src.preprocessing.scaler module
        module = importlib.import_module(f'src.preprocessing.scaler.{scaler_method}')
        scaler_function = getattr(module, 'scaler')
        return scaler_function
    except ModuleNotFoundError as e:
        raise ImportError(f"Module not found for {scaler_method}: {e}")
    except AttributeError as e:
        raise ImportError(f"Function replace_missing_values not found for {scaler_method}: {e}")

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
        # Dynamically import the imputation method from the src.preprocessing.balance module
        module = importlib.import_module(f'src.preprocessing.balance.{balance_method}')
        balance_function = getattr(module, 'balance_train')
        return balance_function
    except ModuleNotFoundError as e:
        raise ImportError(f"Module not found for {balance_method}: {e}")
    except AttributeError as e:
        raise ImportError(f"Function balance_train not found for {balance_method}: {e}")

def dynamic_import_outliers(outliers_method):
    try:
        # Dynamically import the imputation method from the src.preprocessing.scaler module
        module = importlib.import_module(f'src.preprocessing.outliers.{outliers_method}')
        outliers_function = getattr(module, 'remove_outliers')
        return outliers_function
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

def check_raw(dataset):
    try:
        check_raw = read_raw(dataset)
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

def check_external():
    try:
        check_external = read_external()
    except:
        print(f'External không có')
        return False
    return True

def read_external():
    current_dir = Path(__file__).parent
    model_path = current_dir / '..' / 'data' / 'external' / 'external.csv'
    df = pd.read_csv(model_path.resolve())
    return df

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
    scaler_selection = scaler_select()
    try:
        scaler_function = dynamic_import_scaler(scaler_selection)
        X_train_scaled, X_test_scaled = scaler_function(X_train, X_test)
        X_train = pd.DataFrame(X_train_scaled,columns=X_train.columns)
        X_test = pd.DataFrame(X_test_scaled,columns=X_test.columns)
        return X_train, X_test
    except Exception as e:
        print(f"Chưa chọn scaler: {e}")

def joblib_dump(model,name):
    model_path = find_path_from_models(name)
    joblib.dump(model,model_path)

def joblib_load(name):
    model_path = find_path_from_models(name)
    loaded = joblib.load(model_path)
    return loaded
