import pandas as pd
import sqlite3
import os
import glob
import logging
from pdfminer.high_level import extract_text

# Set up basic logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.
    """
    logging.info(f"Extracting text from PDF: {file_path}")
    text = extract_text(file_path)
    logging.info(f"Finished extracting text from PDF: {file_path}")
    return text


def get_pdf_files(directory):
    """
    Gets all pdf files from a directory.
    """
    logging.info(f"Getting PDF files from directory: {directory}")
    files = glob.glob(os.path.join(directory, '*.pdf'))
    logging.info(f"Found {len(files)} PDF files.")
    return files


def get_csv_files(directory):
    """
    Gets all csv files from a directory.
    """
    logging.info(f"Getting CSV files from directory: {directory}")
    files = glob.glob(os.path.join(directory, '*.csv'))
    logging.info(f"Found {len(files)} CSV files.")
    return files

def clean_data(df):
    """
    Cleans the dataframe.
    """
    if df is None or df.empty:
        logging.warning("Dataframe is None or empty, skipping cleaning.")
        return None
    
    logging.info("Cleaning data.")
    # Remove unwanted index column
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    df.dropna(inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        # Use .apply(str) to ensure all values are strings before replacement
        df[col] = df[col].astype(str).apply(lambda x: x.replace('fraud_', '').replace('_', ' ').title())
    
    logging.info("Finished cleaning data.")
    return df

def save_to_db_incremental(df, conn, table_name, first_chunk):
    """
    Saves a dataframe chunk to a sqlite database.
    
    The 'if_exists' mode is set to 'replace' for the first chunk 
    and 'append' for subsequent chunks.
    """
    if df is None or df.empty:
        logging.warning("Dataframe is None or empty, skipping saving to database.")
        return False # Return False if nothing was saved
    
    # Use 'replace' for the first chunk to create/overwrite the table, 
    # and 'append' for all subsequent chunks.
    if_exists_mode = 'replace' if first_chunk else 'append'
    
    logging.info(f"Saving dataframe chunk to database, mode: '{if_exists_mode}'.")
    try:
        df.to_sql(table_name, conn, if_exists=if_exists_mode, index=False)
        return True # Return True if saving was successful
    except Exception as e:
        logging.error(f"Error saving chunk to DB: {e}")
        return False

def create_tab_db():
    """
    Main function to execute the data processing pipeline, 
    processing CSV files one-by-one and incrementally saving to the database.
    """
    # Define paths
    data_dir = 'data'
    media_dir = 'media'
    db_name = 'fraud.db'
    db_path = os.path.join(media_dir, db_name)
    table_name = 'fraud_data'
    
    # Check if DB already exists (to avoid re-running if not needed)
    if os.path.exists(db_path):
        logging.info(f"Database '{db_name}' already exists. Skipping creation.")
        return
    
    # Create media directory if it doesn't exist
    if not os.path.exists(media_dir):
        os.makedirs(media_dir)

    csv_files = get_csv_files(data_dir)
    if not csv_files:
        logging.warning(f"No CSV files found in the '{data_dir}' directory.")
        # Create an empty database to signify completion
        conn_temp = sqlite3.connect(db_path)
        pd.DataFrame().to_sql(table_name, conn_temp, if_exists='replace', index=False)
        conn_temp.close()
        logging.info(f"Empty database '{db_name}' created in '{media_dir}'.")
        return


    logging.info(f"Starting incremental processing of {len(csv_files)} CSV files.")
    conn = sqlite3.connect(db_path)
    first_chunk = True
    total_files_processed = 0

    for file_path in csv_files:
        logging.info(f"Processing file: {file_path}")
        
        try:
            # Read, clean, and save the data chunk
            df_chunk = pd.read_csv(file_path)
            df_chunk = clean_data(df_chunk)
            
            if df_chunk is not None and not df_chunk.empty:
                if save_to_db_incremental(df_chunk, conn, table_name, first_chunk):
                    first_chunk = False  # Next save will be an 'append'
                    total_files_processed += 1
            else:
                 logging.warning(f"File {file_path} resulted in an empty dataframe after cleaning.")

        except Exception as e:
            logging.error(f"Failed to process file {file_path}: {e}")

    conn.close()
    
    if total_files_processed > 0:
        logging.info(f"Successfully processed {total_files_processed} files and saved data to '{db_path}' in table '{table_name}'.")
    else:
        logging.warning(f"No valid data was saved to the database. Check logs for warnings/errors.")
