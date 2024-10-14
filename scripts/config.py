import configparser
import logging

# Load config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Access config values
eeg_file_path = config['Paths']['eeg_file_path']
fif_file_path = config['Paths']['fif_file_path']  # New addition for FIF path
edf_output_path = config['Paths']['edf_output_path']  # New addition for EDF output path
l_freq = config.getfloat('EEGProcessing', 'l_freq')
tmin = config.getfloat('Epochs', 'tmin')
tmax = config.getfloat('Epochs', 'tmax')
baseline = eval(config['Epochs']['baseline'])  # Convert string to tuple

# Example of accessing other settings
ica_n_components = config.getint('EEGProcessing', 'ica_n_components')
ica_max_iter = config.getint('EEGProcessing', 'ica_max_iter')

# Setup logging
log_file = config['Paths']['log_file']
log_level = getattr(logging, config['Logging']['log_level'].upper(), logging.INFO)  # Ensure log level is valid
logging.basicConfig(
    filename=log_file,
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
