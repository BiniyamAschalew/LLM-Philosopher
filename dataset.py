import os
import pandas as pd

def load_csv_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file as a pandas DataFrame.
    
    Args:
    - file_path: str, path to the CSV file.
    
    Returns:
    - pd.DataFrame: the loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)

def load_json_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a JSON file as a pandas DataFrame.
    
    Args:
    - file_path: str, path to the JSON file.
    
    Returns:
    - pd.DataFrame: the loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_json(file_path)

def load_excel_dataset(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """
    Load an Excel file as a pandas DataFrame.
    
    Args:
    - file_path: str, path to the Excel file.
    - sheet_name: str, name of the sheet to load (optional).
    
    Returns:
    - pd.DataFrame: the loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_excel(file_path, sheet_name=sheet_name)

def load_dataset(file_path: str, file_type: str = 'csv', sheet_name: str = None) -> pd.DataFrame:
    """
    General function to load a dataset based on file type.
    
    Args:
    - file_path: str, path to the dataset file.
    - file_type: str, type of the dataset file ('csv', 'json', 'excel').
    - sheet_name: str, name of the sheet to load (for Excel files).
    
    Returns:
    - pd.DataFrame: the loaded DataFrame.
    """
    loaders = {
        'csv': load_csv_dataset,
        'json': load_json_dataset,
        'excel': load_excel_dataset
    }
    
    if file_type not in loaders:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return loaders[file_type](file_path) if file_type != 'excel' else loaders[file_type](file_path, sheet_name)


