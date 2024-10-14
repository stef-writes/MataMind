import os
import mne
import pandas as pd  # Import pandas to create a DataFrame
import logging  # For traceability
import configparser
import matplotlib.pyplot as plt  # For visualization
import numpy as np  # For unique descriptions

# Load config.ini for paths and parameters
config = configparser.ConfigParser()
config.read('config.ini')

# Define paths from config
base_data_dir = '/Users/nooz/MetaViz/MataMind/scripts/physionet.org'
eeg_data_dir = config['Paths']['eeg_data_dir']

# Setup logging
log_file = os.path.join(eeg_data_dir, 'eeg_analysis.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# Updated task mapping with all relevant run IDs
task_mapping = {
    'R01': ('rest', 'rest'),  # Assuming R01 is a rest condition
    'R02': ('rest', 'rest'),  # Assuming R02 is a rest condition
    'R03': ('left fist', 'right fist'),
    'R04': ('left fist', 'right fist'),
    'R07': ('left fist', 'right fist'),
    'R08': ('left fist', 'right fist'),
    'R11': ('left fist', 'right fist'),
    'R12': ('left fist', 'right fist'),
    'R05': ('both fists', 'both feet'),
    'R06': ('both fists', 'both feet'),
    'R09': ('both fists', 'both feet'),
    'R10': ('both fists', 'both feet'),
    'R13': ('both fists', 'both feet'),
    'R14': ('both fists', 'both feet')
    # Add other runs as needed
}

# Function to get task info based on run number
def get_run_info(edf_filepath):
    filename = os.path.basename(edf_filepath)
    run_id = filename.split('R')[-1].split('.')[0]  # Extract run number, e.g., '09'
    task_info = task_mapping.get(f'R{run_id}', ('unknown', 'unknown'))
    logging.info(f"Extracted run info: Run ID = {run_id}, Tasks = {task_info}")
    return task_info

# Iterate over all EDF files in the base data directory
for root, _, files in os.walk(base_data_dir):
    for file in files:
        if file.endswith('.edf'):
            edf_file_path = os.path.join(root, file)

            # Load the raw EEG data from .edf
            logging.info(f"Loading raw EEG data from {edf_file_path}...")
            try:
                raw = mne.io.read_raw_edf(edf_file_path, preload=True)
            except FileNotFoundError:
                logging.error(f"EDF file not found at {edf_file_path}")
                continue
            except Exception as e:
                logging.error(f"Error loading EDF file: {e}")
                continue

            # Extract events and annotations from the raw data
            logging.info("Extracting events and annotations from raw data...")
            try:
                annotations = raw.annotations
                descriptions = annotations.description
                unique_descriptions = np.unique(descriptions)
                logging.info(f"Unique annotations descriptions: {unique_descriptions}")

                # Get run-specific task information
                left_task, right_task = get_run_info(edf_file_path)
                logging.info(f"Task info for this run: T1 = {left_task}, T2 = {right_task}")

                # Map annotations descriptions to integer event codes
                event_id = {'T0': 0}  # Assuming T0 is rest or baseline

                # Initialize condition to event code mapping
                condition_to_event_code = {}
                event_code = 1

                # Map 'T1' to left_task
                if left_task != 'unknown':
                    condition_to_event_code[left_task] = event_code
                    event_id['T1'] = event_code
                    event_code += 1

                # Map 'T2' to right_task
                if right_task != 'unknown':
                    condition_to_event_code[right_task] = event_code
                    event_id['T2'] = event_code
                    event_code += 1

                logging.info(f"Event ID mapping for this run: {event_id}")
                logging.info(f"Condition to Event Code mapping: {condition_to_event_code}")

                # Extract events using the updated event_id mapping
                events, _ = mne.events_from_annotations(raw, event_id=event_id)

            except Exception as e:
                logging.error(f"Error extracting events from {edf_file_path}: {e}")
                continue

            # Create epochs with event IDs corresponding to condition names
            logging.info("Creating epochs for the tasks...")
            try:
                epochs = mne.Epochs(raw, events, event_id=condition_to_event_code,
                                    tmin=-0.2, tmax=1.0, baseline=(None, 0), preload=True)
            except Exception as e:
                logging.error(f"Error creating epochs for {edf_file_path}: {e}")
                continue

            # Assign metadata with the task names
            metadata = pd.DataFrame({
                'task': [left_task if event[2] == condition_to_event_code.get(left_task) else right_task
                         for event in events if event[2] in condition_to_event_code.values()]
            })

            # Assign the metadata DataFrame to the epochs
            epochs.metadata = metadata
            logging.info("Metadata assigned to epochs.")

            # Log the number of epochs for each task
            for condition_name in condition_to_event_code.keys():
                num_epochs = len(epochs[condition_name])
                logging.info(f"Number of epochs for {condition_name}: {num_epochs}")

            # Save the epochs with a dynamic file path
            input_filename = os.path.basename(edf_file_path)
            base_filename = os.path.splitext(input_filename)[0]
            output_filename = f'{base_filename}_epochs-epo.fif'
            output_filepath = os.path.join(eeg_data_dir, output_filename)

            logging.info(f"Saving epochs to {output_filepath}...")
            epochs.save(output_filepath, overwrite=True)
            logging.info("Epoch extraction complete with run-based task segmentation.")

            # ------------------------------
            # ADDITION: Epoch Visualization
            # ------------------------------

            # Plot the epochs
            logging.info("Visualizing the epochs...")
            try:
                epochs.plot(title=f'Epochs for {left_task} and {right_task}', n_epochs=10, n_channels=10)
                plt.show()  # Ensure the plot window stays open
            except Exception as e:
                logging.error(f"Error visualizing epochs for {edf_file_path}: {e}")
                continue
