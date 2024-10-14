import mne
import os
import configparser
import matplotlib.pyplot as plt
import numpy as np

# Load config file
config = configparser.ConfigParser()
config.read('config.ini')

# Dynamically set subject and run IDs
subject_id = 'S001'  # Full ID for folder
run_id = '09'
subject_num = subject_id[1:]  # Numeric part for file name

# Build the EEG file path using the template from config
eeg_file_template = config['Paths']['eeg_file_template']
eeg_file_path = eeg_file_template.format(subject_id=subject_id, subject_num=subject_num, run_id=run_id)

# Get output path for visualizations
visualization_output_path = config['Paths']['visualization_output_path']

# Ensure the visualization directory exists
if not os.path.exists(visualization_output_path):
    os.makedirs(visualization_output_path)

# Function to load the EEG data, either in EDF or FIF format
def load_eeg_data(filepath):
    """Load EEG data from an EDF or FIF file."""
    print(f"\nLoading data from: {filepath}", flush=True)
    if filepath.endswith('.edf'):
        raw = mne.io.read_raw_edf(filepath, preload=True)
    elif filepath.endswith('.fif'):
        raw = mne.io.read_raw_fif(filepath, preload=True)
    else:
        raise ValueError("Unsupported file format. Please provide an EDF or FIF file.")
    return raw

# Rename channels to standard names based on config file
def rename_channels(raw):
    # Get mapping from config
    mapping = dict(config['EEGChannelMapping'])
    print("\nRenaming channels with the following mapping:", mapping, flush=True)  # Debugging line

    # Convert raw channel names and mapping keys to lowercase for consistency
    original_channels = {ch.lower(): ch for ch in raw.info['ch_names']}
    corrected_mapping = {}
    
    # Build corrected mapping if keys match with raw channels
    for key, value in mapping.items():
        if key.lower() in original_channels:
            corrected_mapping[original_channels[key.lower()]] = value

    # Debugging: print out which channels were successfully matched
    print("\nCorrected Mapping Ready for Renaming Channels:", corrected_mapping, flush=True)

    # Check if there are unmatched channels
    unmatched_channels = [ch for ch in raw.info['ch_names'] if ch.lower() not in mapping]
    if unmatched_channels:
        print("\nWarning: The following channels could not be matched and will not be renamed:", unmatched_channels, flush=True)

    # Rename channels using corrected mapping
    raw.rename_channels(corrected_mapping)

# Set the montage from config
def set_montage(raw):
    """Set the montage for electrode positions."""
    montage_type = config['EEGProcessing']['montage_type']
    montage = mne.channels.make_standard_montage(montage_type)
    raw.set_montage(montage)

# Extract events from annotations
def extract_events(raw):
    events, event_ids = mne.events_from_annotations(raw)
    print("\nExtracted Events:", events, flush=True)
    print("\nEvent IDs:", event_ids, flush=True)
    return events, event_ids

# Function to visualize the data
def visualize_data(raw, eeg_file_path):
    # Ensure the output directory exists
    if not os.path.exists(visualization_output_path):
        os.makedirs(visualization_output_path)

    base_filename = os.path.splitext(os.path.basename(eeg_file_path))[0]

    # Plot the raw EEG data
    print("\nPlotting Raw EEG Data...", flush=True)
    raw_fig = raw.plot(show=False)
    raw_fig.savefig(os.path.join(visualization_output_path, f"{base_filename}_raw_eeg.png"))

    # Plot the power spectral density (PSD)
    print("\nPlotting Power Spectral Density (PSD)...", flush=True)
    plt.figure()
    psd_fig = raw.plot_psd(dB=True, show=False)
    psd_fig.savefig(os.path.join(visualization_output_path, f"{base_filename}_psd.png"))

    # Plot the sensor positions
    print("\nPlotting Channel Locations...", flush=True)
    sensors_fig = raw.plot_sensors(show_names=True, show=False)
    sensors_fig.savefig(os.path.join(visualization_output_path, f"{base_filename}_sensor_positions.png"))

# Main preprocessing function
def load_and_preprocess():
    # Load the EEG data
    raw = load_eeg_data(eeg_file_path)

    # Save channel names to a file for inspection
    with open('channel_names.txt', 'w') as f:
        for ch_name in raw.info['ch_names']:
            f.write(f"{ch_name}\n")
    
    # Print channel names to see the actual names in the data
    print("Original channel names saved to 'channel_names.txt'", flush=True)

    # Print the original channel names for debugging
    print("Original channel names:", raw.info['ch_names'], flush=True)

    # Print mapping keys for debugging
    mapping = dict(config['EEGChannelMapping'])
    print("\nMapping keys:", list(mapping.keys()), flush=True)

    # Rename channels
    rename_channels(raw)

    # Set montage
    set_montage(raw)

    # Extract events
    events, event_ids = extract_events(raw)

    # Visualize data (optional, for exploration)
    visualize_data(raw, eeg_file_path)

    return raw, events, event_ids

if __name__ == "__main__":
    raw, events, event_ids = load_and_preprocess()
