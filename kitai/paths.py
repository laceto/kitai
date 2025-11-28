import os 
import logging
from pathlib import Path

# Configure logging once at the application entry point
logging.basicConfig(
    level=logging.INFO,  # switch to DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def check_and_create_folder(folder_path: str) -> Path:
    """
    Ensure that a folder exists at the given path.
    
    Args:
        folder_path (str): Path to the folder to check/create.
    
    Returns:
        Path: A pathlib.Path object pointing to the folder.
    
    Raises:
        ValueError: If folder_path is empty or invalid.
        OSError: If the folder cannot be created due to permissions or other issues.
    """
    if not folder_path or not isinstance(folder_path, str):
        raise ValueError("folder_path must be a non-empty string")

    folder = Path(folder_path).expanduser().resolve()

    if folder.exists():
        if not folder.is_dir():
            raise OSError(f"Path '{folder}' exists but is not a directory.")
        logging.info("Folder '%s' already exists.", folder)
    else:
        try:
            folder.mkdir(parents=True, exist_ok=False)
            logging.info("Folder '%s' created successfully.", folder)
        except Exception as e:
            logging.error("Failed to create folder '%s': %s", folder, e)
            raise

    return folder



def get_file_paths(
    path: str, 
    file_pattern: str
    ) -> list:  
    """  
    Retrieve a list of all file paths in the specified directory and its subdirectories  
    that match the given file pattern.  
  
    This function scans the given directory (and its subdirectories) for files,  
    filters them based on the provided file pattern, and returns a list of their full paths.  
  
    Args:  
        path (str): Directory path where the search for files will be conducted.  
        file_pattern (str): File pattern to filter files (e.g., '.py' for Python files).  
  
    Returns:  
        list: List of full paths to files found in the specified directory that match the pattern.  
  
    Raises:  
        FileNotFoundError: If the specified directory does not exist.  
        Exception: For any other unexpected errors that may occur during file retrieval.  
    """  
    file_paths = []  
    try:  
        # List files in the base directory  
        # base_files = [os.path.join(path, f) for f in os.listdir(path)]  
          
        # Walk through the directory and subdirectories  
        for root, dirs, files in os.walk(path):  
            for file in files:  
                file_path = os.path.join(root, file)  
                file_paths.append(file_path)  
          
        # Filter for files matching the given pattern  
        filtered_file_paths = [file_path for file_path in file_paths if file_path.endswith(file_pattern)]  
        return filtered_file_paths  
  
    except FileNotFoundError as e:  
        print(f"Error: The directory '{path}' does not exist.")  
        raise e  
    except Exception as e:  
        print(f"An unexpected error occurred: {e}")  
        raise e  