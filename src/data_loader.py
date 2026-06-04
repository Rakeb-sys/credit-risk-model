import pandas as pd
import os


def load_data(file_path, **kwargs):

    try:

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load dataset with flexible parameters
        df = pd.read_csv(file_path, **kwargs)

        # Validate dataframe
        if df.empty:
            raise ValueError("Loaded dataframe is empty.")

        print("Dataset loaded successfully.")
        print(f"Shape: {df.shape}")

        return df

    except FileNotFoundError as fnf_error:
        print(f"ERROR: {fnf_error}")

    except pd.errors.EmptyDataError:
        print("ERROR: CSV file is empty.")

    except pd.errors.ParserError:
        print("ERROR: Failed to parse CSV.")

    except Exception as e:
        print(f"Unexpected Error: {e}")

    return None
