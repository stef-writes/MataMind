import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import configparser

# Load config file
config = configparser.ConfigParser()
config.read('config.ini')

# Dynamically set subject and run IDs
subject_id = 'S001'
run_id = '09'

# Build the EEG file path using the template from config
eeg_file_template = config['Paths']['edf_output_template']
eeg_file_path = eeg_file_template.format(subject_id=subject_id, run_id=run_id)

# Define the dataset directory and load the EDF file
def load_eeg_data(filepath):
    print(f"\nLoading data from: {filepath}")
    raw = mne.io.read_raw_edf(filepath, preload=True)
    return raw

def rename_channels(raw):
    # Create a mapping dictionary to rename channels
    mapping = {
        'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCz', 'Fc2.': 'FC2', 'Fc4.': 'FC4', 'Fc6.': 'FC6',
        'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1', 'Cz..': 'Cz', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6',
        'Cp5.': 'CP5', 'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPz', 'Cp2.': 'CP2', 'Cp4.': 'CP4', 'Cp6.': 'CP6',
        'Fp1.': 'Fp1', 'Fpz.': 'Fpz', 'Fp2.': 'Fp2',
        'Af7.': 'AF7', 'Af3.': 'AF3', 'Afz.': 'AFz', 'Af4.': 'AF4', 'Af8.': 'AF8',
        'F7..': 'F7', 'F5..': 'F5', 'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'Fz', 'F2..': 'F2', 'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8',
        'Ft7.': 'FT7', 'Ft8.': 'FT8',
        'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10',
        'Tp7.': 'TP7', 'Tp8.': 'TP8',
        'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1', 'Pz..': 'Pz', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6', 'P8..': 'P8',
        'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POz', 'Po4.': 'PO4', 'Po8.': 'PO8',
        'O1..': 'O1', 'Oz..': 'Oz', 'O2..': 'O2', 'Iz..': 'Iz'
    }
    raw.rename_channels(mapping)

def set_montage(raw):
    # Load a standard montage, such as the 10-20 system
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

def extract_events(raw):
    # Extract events from annotations
    events, event_ids = mne.events_from_annotations(raw)
    print("\nExtracted Events:")
    print(events)
    print("\nEvent IDs:")
    print(event_ids)
    return events, event_ids

def calculate_summary_statistics(raw):
    print("\nBasic summary statistics for each channel:")
    data = raw.get_data()  # Get the data as a numpy array (channels x samples)
    channel_names = raw.ch_names
    for i, channel in enumerate(channel_names):
        mean = np.mean(data[i])
        std_dev = np.std(data[i])
        min_val = np.min(data[i])
        max_val = np.max(data[i])
        print(f"{channel}: Mean = {mean:.2e}, Std Dev = {std_dev:.2e}, Min = {min_val:.2e}, Max = {max_val:.2e}")

def main():
    # Load the EEG data from the dynamically created file path
    raw = load_eeg_data(eeg_file_path)

    # Rename channels to match the montage
    rename_channels(raw)

    # Set the montage to ensure channel positions are correct
    set_montage(raw)

    # Print basic info of the loaded data
    print("\nBasic Info of the Loaded Data:")
    print(raw.info)

    # Additional information to provide a more detailed summary
    print("\nChannel Names:")
    print(raw.ch_names)
    print("\nSampling Frequency:")
    print(f"{raw.info['sfreq']} Hz")
    print("\nMeasurement Date:")
    print(raw.info['meas_date'])
    print("\nData Shape:")
    print(raw._data.shape)
    print("\nDuration of Recording:")
    duration = len(raw.times) / raw.info['sfreq']
    print(f"{duration:.2f} seconds")
    print("\nAvailable Annotations:")
    print(raw.annotations)

    # Extract and print event information
    events, event_ids = extract_events(raw)

    # Calculate and print summary statistics for each channel
    calculate_summary_statistics(raw)

    # Generate dynamic output filenames based on EEG_FILE_PATH
    base_filename = os.path.splitext(os.path.basename(eeg_file_path))[0]

    # Plot the raw EEG data
    print("\nPlotting Raw EEG Data...")
    raw_fig = raw.plot(show=False)
    raw_fig.savefig(f"{base_filename}_raw_eeg.png")

    # Plot the power spectral density (PSD)
    print("\nPlotting Power Spectral Density (PSD)...")
    plt.figure()
    psd_fig = raw.plot_psd(dB=True, show=False)
    psd_fig.savefig(f"{base_filename}_psd.png")

    # Plot the sensor positions
    print("\nPlotting Channel Locations...")
    sensors_fig = raw.plot_sensors(show_names=True, show=False)
    sensors_fig.savefig(f"{base_filename}_sensor_positions.png")

if __name__ == "__main__":
    main()
