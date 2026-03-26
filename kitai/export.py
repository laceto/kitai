import logging

import pandas as pd

logger = logging.getLogger(__name__)


def df_to_csv(
    df: pd.DataFrame,
    path: str
) -> None:
    """
    Write a DataFrame to a CSV file.

    Parameters:
        df (pd.DataFrame): DataFrame to write.
        path (str): Path of the CSV file to be created or overwritten.

    Returns:
        None
    """
    try:
        df.to_csv(path, index=False)
        logger.info("DataFrame successfully written to '%s'.", path)
    except Exception as e:
        logger.error("Error writing CSV to '%s': %s", path, e)
        raise


def df_to_excel(
    df: pd.DataFrame,
    path: str,
    sheet: str = 'Sheet1'
) -> None:
    """
    Write a DataFrame to an Excel file.

    Parameters:
        df (pd.DataFrame): DataFrame to write.
        path (str): Path of the Excel file to be created or overwritten.
        sheet (str): Name of the sheet in the Excel file (default: 'Sheet1').

    Returns:
        None
    """
    try:
        df.to_excel(path, sheet_name=sheet, index=False)
        logger.info("DataFrame successfully written to '%s' in sheet '%s'.", path, sheet)
    except Exception as e:
        logger.error("Error writing Excel to '%s': %s", path, e)
        raise
