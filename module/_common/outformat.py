import pandas as pd
import numpy as np
def format_dataframe_for_export(df:pd.DataFrame, scientific_cols=None, float_cols=None):
    """
    Parameters:
    - df: raw DataFrame
    - scientific_cols: 科学计数法列
    - float_cols: 浮点数列
    """
    df_export = df.copy()
    # Scientific
    if scientific_cols:
        for col in scientific_cols:
            if col in df_export.columns and df_export[col].dtype in [np.float64, np.int64, np.float32]:
                df_export[col] = df_export[col].apply(lambda x: f"{x:.4e}")
    # Float
    if float_cols:
        for col in float_cols:
            if col in df_export.columns and df_export[col].dtype in [np.float64, np.float32]:
                df_export[col] = df_export[col].apply(lambda x: f"{x:.4f}")
    return df_export