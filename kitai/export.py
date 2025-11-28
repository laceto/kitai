import pandas as pd

def df_to_csv(
    df: pd.DataFrame, 
    path: str
    ) -> None:  
    """  
    Write a DataFrame to a csv file.  
  
    Parameters:  
    df (pd.DataFrame): DataFrame to write to Excel.  
    path (str): Path of the Excel file to be created or overwritten.  
    sheet (str): Name of the sheet in the Excel file (default is 'Sheet1').  
  
    Returns:  
    None  
    """  
    try:  
        # Write the DataFrame to an Excel file  
        df.to_csv(path, index=False)  
        print(f"DataFrame successfully written to {path}'.")  
    except Exception as e:  
        print(f"An error occurred while writing to Excel: {e}")  

def df_to_excel(
    df: pd.DataFrame, 
    path: str, 
    sheet: str = 'Sheet1'
    ) -> None:  
    """  
    Write a DataFrame to an Excel file.  
  
    Parameters:  
    df (pd.DataFrame): DataFrame to write to Excel.  
    path (str): Path of the Excel file to be created or overwritten.  
    sheet (str): Name of the sheet in the Excel file (default is 'Sheet1').  
  
    Returns:  
    None  
    """  
    try:  
        # Write the DataFrame to an Excel file  
        df.to_excel(path, sheet_name=sheet, index=False)  
        print(f"DataFrame successfully written to {path} in sheet '{sheet}'.")  
    except Exception as e:  
        print(f"An error occurred while writing to Excel: {e}")  