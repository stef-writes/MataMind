import mne
import pyedflib
import numpy as np
import logging
from config import eeg_file_path, fif_file_path, edf_output_path  # Import paths from config

# Load the .fif file
logging.info(f"Loading .fif file from {fif_file_path}")
raw = mne.io.read_raw_fif(fif_file_path, preload=True)

# Get data and channel info
data = raw.get_data() * 1e6  # Convert from V to µV
n_channels = raw.info['nchan']
sampling_rate = raw.info['sfreq']
channel_names = raw.info['ch_names']

# Create an EDF writer
logging.info(f"Converting and saving as EDF at {edf_output_path}")
edf_writer = pyedflib.EdfWriter(
    edf_output_path,
    n_channels=n_channels,
    file_type=pyedflib.FILETYPE_EDFPLUS
)

# Create channel info for each channel
channel_info = []
for ch in range(n_channels):
    ch_dict = {
        'label': channel_names[ch],
        'dimension': 'uV',
        'sample_rate': sampling_rate,
        'physical_min': -100000,
        'physical_max': 100000,
        'digital_min': -32768,
        'digital_max': 32767,
        'transducer': '',
        'prefilter': ''
    }
    channel_info.append(ch_dict)

# Set channel info and write the data
edf_writer.setSignalHeaders(channel_info)
edf_writer.writeSamples(data)
edf_writer.close()

logging.info("Conversion to EDF completed successfully!")
