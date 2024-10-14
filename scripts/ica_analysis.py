import os
import mne
import logging
import configparser
import numpy as np  # Import numpy for power-based artifact detection
import matplotlib.pyplot as plt

# Load config.ini for paths and parameters
config = configparser.ConfigParser()
config.read('config.ini')

# Define paths from config
eeg_data_dir = config['Paths']['eeg_data_dir']
log_file = os.path.join(eeg_data_dir, 'ica_analysis.log')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# Function to load filtered data (epochs), rename channels, and set montage
def load_filtered_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    logging.info(f"Loading filtered EEG epochs from {filepath}...")
    epochs = mne.read_epochs(filepath, preload=True)
    logging.info("\nChannel Info after loading epochs: %s", epochs.ch_names)

    # Rename channels based on config file
    rename_channels(epochs)

    # Set the montage based on config file
    montage_type = config['EEGProcessing']['montage_type']
    logging.info(f"Setting montage: {montage_type}")
    montage = mne.channels.make_standard_montage(montage_type)
    epochs.set_montage(montage)

    return epochs

# Rename channels to standard names based on config file
def rename_channels(epochs):
    mapping = dict(config['EEGChannelMapping'])
    # Convert mapping keys to lowercase to ensure case-insensitivity
    mapping = {k.lower(): v for k, v in mapping.items()}

    # Get original channel names from the epochs
    original_channels = epochs.ch_names
    new_mapping = {}

    # Create a new mapping based on the original channel names
    for ch in original_channels:
        ch_lower = ch.lower()
        if ch_lower in mapping:
            new_mapping[ch] = mapping[ch_lower]

    logging.info("Renaming channels with the following mapping: %s", new_mapping)
    epochs.rename_channels(new_mapping)

# Apply ICA to identify and remove artifacts
def apply_ica(epochs, n_components=15, random_state=97, max_iter=500):
    logging.info("Applying ICA for artifact analysis...")
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, max_iter=max_iter)
    ica.fit(epochs)
    return ica

# Automatically identify artifacts using power in low frequencies
def auto_identify_artifacts(ica, epochs, threshold=0.15):
    logging.info("Attempting to automatically identify artifacts using power-based detection...")

    # Use power-based artifact detection
    psd = epochs.compute_psd(fmin=0.1, fmax=4.0).get_data()
    high_power_components = [idx for idx in range(psd.shape[0]) if np.max(psd[idx]) > threshold]
    
    logging.info(f"Components with high power in low-frequency range: {high_power_components}")
    return high_power_components

# Visualize ICA components and time courses
def visualize_ica_components(ica, epochs):
    logging.info("Visualizing ICA components...")

    # Plot the spatial patterns of the ICA components
    ica.plot_components(inst=epochs)  # Plot spatial patterns
    plt.show()

    # Plot the time series of the ICA components
    ica.plot_sources(epochs)  # Plot the IC time courses
    plt.show()

    # Plot detailed properties (e.g., power spectrum) of individual ICA components
    ica.plot_properties(epochs, picks=range(15))  # You can select the number of components to plot
    plt.show()

# Clean and save the data
def clean_and_save_data(ica, epochs, exclude, output_filepath):
    if len(exclude) == 0:
        logging.info("No significant artifacts detected. The data appears clean.")
    else:
        logging.info(f"Excluding {len(exclude)} component(s) identified as artifacts: {exclude}")

    # Apply ICA exclusion
    ica.exclude = exclude
    logging.info("Applying ICA to remove artifacts (if any)...")
    epochs_cleaned = ica.apply(epochs.copy())

    # Plot cleaned EEG data
    logging.info("Plotting cleaned EEG data...")
    epochs_cleaned.plot(n_channels=10, title='Cleaned EEG Epochs', show=True)

    # Save cleaned data
    logging.info(f"Saving cleaned EEG epochs to {output_filepath}...")
    epochs_cleaned.save(output_filepath, overwrite=True)
    logging.info("ICA analysis and data cleaning complete.")

# Main function to execute the analysis
def main():
    # Iterate over all epoch files in the data directory
    for root, _, files in os.walk(eeg_data_dir):
        for file in files:
            if file.endswith('_epochs-epo.fif'):
                epochs_file_path = os.path.join(root, file)
                cleaned_filepath = os.path.join(root, f'cleaned_{file}')
                
                # Load the preprocessed epochs data, rename channels, and set montage
                try:
                    epochs = load_filtered_data(epochs_file_path)
                except FileNotFoundError as e:
                    logging.error(e)
                    continue

                # Apply ICA for artifact removal
                ica = apply_ica(epochs)

                # Visualize ICA components and their time series
                visualize_ica_components(ica, epochs)

                # Automatically identify artifacts (using power-based detection since no EOG channels)
                artifacts_to_exclude = auto_identify_artifacts(ica, epochs)

                # Clean and save the data after artifact removal
                clean_and_save_data(ica, epochs, artifacts_to_exclude, cleaned_filepath)

                logging.info(f"ICA analysis, artifact removal, and visualization of cleaned EEG complete for {file}.")

if __name__ == "__main__":
    main()
