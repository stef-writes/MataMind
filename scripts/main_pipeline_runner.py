import os
import sys
import subprocess
import logging
import time
from datetime import datetime

# Configure logging
log_file = "pipeline_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)

# Add the current directory to the system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

logging.info(f"Current working directory: {os.getcwd()}")

# Directory where the cleaned epochs files are saved
directory = '/Users/nooz/MetaViz/MataMind/scripts/'

# Function to dynamically find the latest file
def get_latest_file(extension):
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    if not files:
        raise FileNotFoundError(f"No files with extension {extension} found.")
    latest_file = os.path.join(directory, max(files, key=os.path.getmtime))
    return latest_file

# List of scripts to run sequentially
scripts_to_run = [
    "./load_and_preprocess.py",
    "./extract_epochs.py",
    "./ica_analysis.py",
    "./erp_analysis.py",
    "./psd_analysis.py",
    "./visualizations.py",
    "./source_localization.py"
]

# Function to run each script and capture output
def run_script(script_name, args=[]):
    command = ['python', script_name] + args
    logging.info(f"--- Start running script: {script_name} ---")
    start_time = time.time()

    result = subprocess.run(command, capture_output=True, text=True)
    execution_time = time.time() - start_time

    logging.info(result.stdout)
    if result.returncode != 0:
        logging.error(result.stderr)
        raise RuntimeError(f"Script {script_name} failed to run.")
    
    logging.info(f"--- Finished running script: {script_name} in {execution_time:.2f} seconds ---")

# Function to check for dependencies before running a script
def check_dependencies(script):
    if script == "./extract_epochs.py":
        filtered_file = get_latest_file('_filtered_raw.fif')
        if not os.path.exists(filtered_file):
            raise RuntimeError(f"Filtered EEG data ({filtered_file}) not found. Cannot run extract_epochs.py")
    elif script == "./ica_analysis.py":
        epochs_file = get_latest_file('_epochs-epo.fif')
        if not os.path.exists(epochs_file):
            raise RuntimeError(f"Epochs data ({epochs_file}) not found. Cannot run ica_analysis.py")
    elif script == "./erp_analysis.py" or script == "./psd_analysis.py":
        # Updated to reflect the correct cleaned EEG file name
        ica_cleaned_file = os.path.join(directory, 'cleaned_subject_raw.fif')
        if not os.path.exists(ica_cleaned_file):
            raise RuntimeError(f"ICA cleaned data ({ica_cleaned_file}) not found. Cannot run {script}")

# Function to run additional analysis and visualization
def run_analysis_and_visualization():
    import mne
    import pyvista as pv  # PyVista for 3D visualization

    # Get the latest cleaned raw and epochs file
    cleaned_raw_file = os.path.join(directory, 'cleaned_subject_raw.fif')
    epochs_file = get_latest_file('_epochs-epo.fif')

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Load Cleaned Raw Data ---
    print(f"Loading cleaned EEG data from {cleaned_raw_file}...")
    try:
        raw = mne.io.read_raw_fif(cleaned_raw_file, preload=True)
    except FileNotFoundError:
        print(f"Error: File {cleaned_raw_file} not found.")
        return
    except Exception as e:
        print(f"Error loading raw file: {e}")
        return

    # (Add the rest of your processing and visualization code here)

    print("Script completed successfully.")

# Main function to run the pipeline
def main():
    start_time = datetime.now()
    logging.info("Pipeline execution started.")
    
    try:
        for script in scripts_to_run:
            check_dependencies(script)
            run_script(script)

        # Run the new analysis and visualization function
        run_analysis_and_visualization()

    except RuntimeError as e:
        logging.error(f"Pipeline execution stopped due to an error: {e}")
    else:
        logging.info("Pipeline executed successfully.")
    finally:
        total_duration = datetime.now() - start_time
        logging.info(f"Total pipeline execution time: {total_duration}")

# Run the pipeline
if __name__ == "__main__":
    main()
