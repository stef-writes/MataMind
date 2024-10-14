"""
psd_analysis.py

This script computes and visualizes the Power Spectral Density (PSD) of the cleaned EEG data for a selected subject, highlighting frequency components.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import configparser
import logging

# Load config.ini for paths and parameters
config = configparser.ConfigParser()
config.read('config.ini')

# Setup logging
log_file = os.path.join(config['Paths']['eeg_data_dir'], 'psd_analysis.log')
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

# Find the latest cleaned epochs file dynamically
files = [f for f in os.listdir(eeg_data_dir) if f.endswith('_epochs-epo.fif')]
if not files:
    logging.error("No cleaned epochs file found in the specified directory.")
    raise FileNotFoundError("No cleaned epochs file found.")

# Use the first available file (or adjust logic if a specific file is needed)
cleaned_epochs_filepath = os.path.join(eeg_data_dir, files[0])
logging.info(f"Loading cleaned EEG data from {cleaned_epochs_filepath}...")

# Load the cleaned EEG data from the most recent file
epochs = mne.read_epochs(cleaned_epochs_filepath, preload=True)
raw_data = epochs.average().copy()  # Average epochs to compute PSD on averaged data

# Optionally select specific channels (e.g., frontal channels) from config
selected_channels = config['PSDAnalysis']['selected_channels'].split(', ')
if any(ch in raw_data.ch_names for ch in selected_channels):
    raw_data.pick_channels(selected_channels)
    logging.info(f"Selected channels for analysis: {selected_channels}")
else:
    logging.info("Using all channels for analysis.")

# Power Spectral Density Calculation
logging.info("Calculating Power Spectral Density (PSD) using multitaper method...")
psd_method = config['PSDAnalysis']['psd_method']
fmin = float(config['PSDAnalysis']['fmin'])
fmax = float(config['PSDAnalysis']['fmax'])

psd, freqs = raw_data.compute_psd(method=psd_method, fmin=fmin, fmax=fmax).get_data(return_freqs=True)
psd_db = 10 * np.log10(psd)  # Convert power to dB

# Plotting the PSD
plt.figure(figsize=(12, 8))
mean_psd = np.mean(psd_db, axis=0)  # Average over channels
std_psd = np.std(psd_db, axis=0)  # Standard deviation across channels

plt.plot(freqs, mean_psd, label='Average PSD', color='b')
plt.fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd, color='b', alpha=0.3, label='±1 SD')

# Mark specific frequency bands
bands = {
    'Delta (1-4 Hz)': (1, 4),
    'Theta (4-8 Hz)': (4, 8),
    'Alpha (8-12 Hz)': (8, 12),
    'Beta (12-30 Hz)': (12, 30),
    'Gamma (30-50 Hz)': (30, 50)
}
for band, (fmin_band, fmax_band) in bands.items():
    plt.axvspan(fmin_band, fmax_band, color='gray', alpha=0.2, label=f'{band} Band')

# Plot labels and title
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('Average Power Spectral Density')
plt.legend()

# Save the plot as an image file with a dynamic name
base_filename = os.path.splitext(os.path.basename(cleaned_epochs_filepath))[0]
output_image = os.path.join(visualization_output_path, f'{base_filename}_psd_plot.png')
plt.savefig(output_image, dpi=300)
logging.info(f"PSD plot saved as {output_image}")

# Show the plot
plt.show()

logging.info("PSD analysis complete.")
