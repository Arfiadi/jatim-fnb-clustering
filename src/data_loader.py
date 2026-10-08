"""
Data loading and basic cleaning module.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def get_default_data_path() -> Path:
    """Find and return the path to the East Java expenditure dataset."""
    base_dir = Path(__file__).resolve().parent.parent
    possible_paths = [
        base_dir / "data" / "raw" / "pengeluaran_jatim_2024.xlsx",
        base_dir / "Rata-rata Pengeluaran Perkapita Seminggu  Menurut Kelompok Makanan Minuman Jadi di Jawa Timur, 2024.xlsx",
        base_dir / "data" / "raw" / "pengeluaran_jatim_2024.csv",
    ]
    for path in possible_paths:
        if path.exists():
            return path
            
    # Search for any .xlsx in project directory
    xlsx_files = list(base_dir.glob("*.xlsx")) + list((base_dir / "data" / "raw").glob("*.xlsx"))
    if xlsx_files:
        return xlsx_files[0]
        
    raise FileNotFoundError("Dataset pengeluaran Jawa Timur tidak ditemukan.")


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load the raw dataset from Excel or CSV file.
    
    Args:
        file_path: Optional explicit file path. If None, uses default search.
        
    Returns:
        pd.DataFrame containing raw survey data.
    """
    path = Path(file_path) if file_path else get_default_data_path()
    if path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=0)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Format file '{path.suffix}' tidak didukung.")
    return df


def clean_data(
    df_raw: pd.DataFrame, 
    region_col: str = "Kabupaten/Kota"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Clean data by isolating region names and preparing numeric features.
    
    Args:
        df_raw: Raw DataFrame loaded from Excel.
        region_col: Name of column containing region/district names.
        
    Returns:
        Tuple containing:
            - df_clean: DataFrame with Kabupaten/Kota and numeric columns (strings like '-' replaced).
            - regions: Series of Kabupaten/Kota names.
            - df_numeric: DataFrame of purely numeric candidate features before dropping columns.
    """
    df = df_raw.copy()
    if region_col not in df.columns:
        raise KeyError(f"Kolom '{region_col}' tidak ditemukan dalam dataset.")
        
    regions = df[region_col].copy()
    data_features = df.drop(columns=[region_col])
    
    # Replace '-' with NaN and coerce all columns to float
    for col in data_features.columns:
        if data_features[col].dtype == object:
            data_features[col] = data_features[col].astype(str).str.strip().replace({"-": np.nan, "": np.nan})
            data_features[col] = pd.to_numeric(data_features[col], errors="coerce")
            
    df_clean = pd.concat([regions, data_features], axis=1)
    return df_clean, regions, data_features


def get_numeric_features(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """Extract list of numeric feature column names.
    
    Args:
        df: Input DataFrame.
        exclude_cols: List of column names to exclude (e.g. labels, clusters).
        
    Returns:
        List of numeric column names.
    """
    exclude = set(exclude_cols or [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]
