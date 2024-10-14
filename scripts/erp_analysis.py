import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import configparser
import logging

# Load config.ini for paths and parameters
config = configparser.ConfigParser()
config.read('config.ini')

# Setup logging
log_file = os.path.join(config['Paths']['eeg_data_dir'], 'erp_analysis.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Define paths from config
eeg_data_dir = config['Paths']['eeg_data_dir']
visualization_output_path = config['Paths']['visualization_output_path']

# Function to load epochs data
def load_epochs(filepath):
    logging.info(f"Loading cleaned epochs data from {filepath}...")
    return mne.read_epochs(filepath, preload=True)

# Standardize channel names to match montage naming conventions
def standardize_channel_names(epochs):
    rename_mapping = dict(config['EEGChannelMapping'])
    rename_mapping = {k.lower(): v for k, v in rename_mapping.items()}
    original_channels = epochs.ch_names
    new_mapping = {ch: rename_mapping[ch.lower()] for ch in original_channels if ch.lower() in rename_mapping}

    logging.info("Renaming channels with the following mapping: %s", new_mapping)
    epochs.rename_channels(new_mapping)

# Diagnostic: Print event IDs and epoch counts
def print_epoch_info(epochs, conditions, event_mapping):
    logging.info(f"Event IDs: {epochs.event_id}")
    for condition in conditions:
        mapped_condition = event_mapping.get(condition, None)
        if mapped_condition in epochs.event_id:
            logging.info(f"Number of epochs for {condition}: {len(epochs[mapped_condition])}")
        else:
            logging.warning(f"Condition '{condition}' not found in event IDs.")

# Plot ERP for each event type and save
def plot_erps(epochs, conditions, event_mapping, base_filename):
    logging.info("Plotting butterfly plots for motor imagery tasks...")
    for condition in conditions:
        mapped_condition = event_mapping.get(condition, None)
        if mapped_condition in epochs.event_id:
            evoked_epochs = epochs[mapped_condition]
            if len(evoked_epochs) > 0:
                logging.info(f"Plotting butterfly plot for condition: {condition}")
                evoked = evoked_epochs.average()
                fig = evoked.plot(show=False, title=f'ERP for {condition}')
                
                # Save the plot instead of displaying it
                output_path = os.path.join(visualization_output_path, f"{base_filename}_{condition}_butterfly.png")
                fig.savefig(output_path)
                plt.close(fig)
            else:
                logging.warning(f"No valid epochs found for condition: {condition}.")
        else:
            logging.warning(f"Skipping plotting for '{condition}' as it is not available in the epochs.")

# Plot overlay of Right Hand vs Left Hand ERP comparison and save
def plot_erp_comparison(epochs, conditions, event_mapping, base_filename):
    mapped_conditions = [event_mapping.get(cond, None) for cond in conditions]
    if all(cond in epochs.event_id for cond in mapped_conditions):
        right_epochs = epochs[mapped_conditions[0]]
        left_epochs = epochs[mapped_conditions[1]]

        if len(right_epochs) > 0 and len(left_epochs) > 0:
            logging.info("Plotting ERP comparison for motor imagery tasks...")
            evoked_dict = {
                'Right Hand': right_epochs.average(),
                'Left Hand': left_epochs.average()
            }
            try:
                fig, ax = plt.subplots()
                mne.viz.plot_compare_evokeds(evoked_dict, picks='eeg', axes=ax, show=False, title="ERP Comparison: Right Hand vs Left Hand")
                
                # Save the comparison plot
                output_path = os.path.join(visualization_output_path, f"{base_filename}_erp_comparison.png")
                fig.savefig(output_path)
                plt.close(fig)
            except ValueError as e:
                logging.error(f"Error while plotting ERP comparison: {e}")
        else:
            logging.warning("Not enough valid epochs to plot ERP comparison for both conditions.")
    else:
        logging.warning("One or more conditions are missing from the event IDs. Skipping ERP comparison.")

# Main function to execute the ERP analysis
def main():
    # Find the latest cleaned epochs file dynamically
    files = [f for f in os.listdir(eeg_data_dir) if f.endswith('_epochs-epo.fif')]
    if not files:
        logging.error("No cleaned epochs file found in the specified directory.")
        return

    # Use the first file found instead of relying on modification time
    epochs_filepath = os.path.join(eeg_data_dir, files[0])

    # Extract the base filename for use in saved plot filenames
    base_filename = os.path.splitext(os.path.basename(epochs_filepath))[0]

    # Conditions for ERP analysis
    conditions = ['Right_Hand', 'Left_Hand']
    event_mapping = {
        'Right_Hand': int(config['Conditions']['T1_event_id']),
        'Left_Hand': int(config['Conditions']['T2_event_id'])
    }

    # Load epochs
    try:
        epochs = load_epochs(epochs_filepath)
    except FileNotFoundError:
        logging.error(f"Could not load the epochs file: {epochs_filepath}")
        return

    # Standardize channel names
    standardize_channel_names(epochs)

    # Set montage to include channel coordinates
    montage_type = config['EEGProcessing']['montage_type']
    logging.info(f"Setting montage: {montage_type}")
    montage = mne.channels.make_standard_montage(montage_type)
    epochs.set_montage(montage, on_missing='ignore')
    logging.info("Montage successfully applied for channel locations.")

    # Print diagnostic information about the epochs
    print_epoch_info(epochs, conditions, event_mapping)

    # Plot ERPs for each condition and save
    plot_erps(epochs, conditions, event_mapping, base_filename)

    # Plot ERP comparison between conditions and save
    plot_erp_comparison(epochs, conditions, event_mapping, base_filename)

    logging.info("ERP analysis complete.")

if __name__ == "__main__":
    main()
