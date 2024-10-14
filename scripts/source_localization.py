import os
import mne
import numpy as np
import pyvista as pv  # PyVista for 3D visualization

# --- Configuration ---
directory = '/Users/nooz/MetaViz/MataMind/scripts/eeg_data/'

# Find the latest cleaned file dynamically
cleaned_files = [
    f for f in os.listdir(directory)
    if f.startswith('cleaned_') and f.endswith('_epochs-epo.fif') and os.path.isfile(os.path.join(directory, f))
]

# Ensure files exist before getting the latest
valid_files = [f for f in cleaned_files if os.path.exists(os.path.join(directory, f))]

if not valid_files:
    print("Error: No valid cleaned EEG files found.")
    exit()

# Pick the latest cleaned file
cleaned_raw_file = os.path.join(directory, max(valid_files, key=lambda f: os.path.getmtime(os.path.join(directory, f))))
epochs_file = cleaned_raw_file  # Use the same file for epochs in this case
output_dir = "output"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate a dynamic base filename for outputs
base_filename = os.path.splitext(os.path.basename(cleaned_raw_file))[0]

# --- Load Cleaned Epochs Data ---
print(f"Loading cleaned EEG epochs data from {cleaned_raw_file}...")
try:
    epochs = mne.read_epochs(cleaned_raw_file, preload=True)
    print("Epochs data loaded successfully.")
    print(epochs)
except FileNotFoundError:
    print(f"Error: File {cleaned_raw_file} not found.")
    exit()
except Exception as e:
    print(f"Error loading epochs file: {e}")
    exit()

# --- Check and Process Channel Names ---
print("Channel names before any renaming:")
print(epochs.info['ch_names'])
montage = mne.channels.make_standard_montage('standard_1020')
expected_ch_names = set(montage.ch_names)
current_ch_names_epochs = set(epochs.info['ch_names'])

missing_ch_names_epochs = expected_ch_names - current_ch_names_epochs
extra_ch_names_epochs = current_ch_names_epochs - expected_ch_names

print(f"Channels missing from epochs (but present in montage): {missing_ch_names_epochs}")
print(f"Extra channels in epochs (not present in montage): {extra_ch_names_epochs}")

# --- Renaming channels to match standard 10-20 system ---
mapping_epochs = {ch_name: ch_name.rstrip('.') for ch_name in epochs.info['ch_names']}
epochs.rename_channels(mapping_epochs)

print("Renaming channels in epochs...")
print(f"Epoch channels after renaming: {epochs.info['ch_names']}")

# --- Drop extra channels not in the montage ---
extra_ch_names_epochs = set(epochs.info['ch_names']) - expected_ch_names
if extra_ch_names_epochs:
    print(f"Dropping extra channels from epochs: {extra_ch_names_epochs}")
    epochs.drop_channels(list(extra_ch_names_epochs))
else:
    print("No extra channels to drop from epochs.")

# --- Set Montage and Reference ---
epochs.set_montage(montage, match_case=False)
print("Setting montage to standard 1020 system...")
epochs.set_eeg_reference('average', projection=True)
epochs.apply_proj()

# --- Fetch fsaverage Template ---
print("Fetching fsaverage template...")
try:
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
    subjects_dir = os.path.dirname(fs_dir)
    os.environ['SUBJECTS_DIR'] = subjects_dir
except Exception as e:
    print(f"Error fetching fsaverage template: {e}")
    exit()

# --- Set Up Source Space ---
print("Setting up source space...")
try:
    src = mne.setup_source_space('fsaverage', spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
except Exception as e:
    print(f"Error setting up source space: {e}")
    exit()

# --- Create BEM Model and Solution ---
print("Creating BEM model...")
try:
    model = mne.make_bem_model(subject='fsaverage', ico=4, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
except Exception as e:
    print(f"Error creating BEM model: {e}")
    exit()

# --- Compute Forward Solution ---
print("Computing forward solution...")
try:
    trans = 'fsaverage'
    fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=1)
except Exception as e:
    print(f"Error computing forward solution: {e}")
    exit()

# --- Compute Noise Covariance ---
print("Computing noise covariance...")
try:
    noise_cov = mne.compute_covariance(epochs, tmax=0., method='shrunk', rank=None)
except Exception as e:
    print(f"Error computing noise covariance: {e}")
    exit()

# --- Compute Inverse Operator ---
print("Computing inverse operator...")
try:
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, loose=0.2, depth=0.8)
except Exception as e:
    print(f"Error computing inverse operator: {e}")
    exit()

# --- Separate Epochs by Condition ---
print("Separating epochs by condition...")
conditions = list(epochs.event_id.keys())
print(f"Conditions found: {conditions}")

epochs_conditions = {}
for condition in conditions:
    epochs_conditions[condition] = epochs[condition]
    print(f"Number of epochs for {condition}: {len(epochs_conditions[condition])}")

# --- Compute Source Estimates for Each Condition ---
print("Computing source estimates for each condition...")
lambda2 = 1.0 / 9.0
method = 'dSPM'

stcs_conditions = {}
for condition in conditions:
    print(f"Processing condition: {condition}")
    stcs_conditions[condition] = mne.minimum_norm.apply_inverse_epochs(
        epochs_conditions[condition], inverse_operator, lambda2, method=method, verbose=True
    )

# --- Compute Average Source Estimates for Each Condition ---
print("Computing average source estimates for each condition...")
stc_averages = {}
for condition in conditions:
    stcs = stcs_conditions[condition]
    stc_average = sum(stcs) / len(stcs)
    stc_averages[condition] = stc_average

# --- Visualize Differences Between Conditions ---
# Example: Difference between 'left fist' and 'right fist'
if 'left fist' in stc_averages and 'right fist' in stc_averages:
    print("Computing difference between 'left fist' and 'right fist'...")
    stc_difference = stc_averages['left fist'] - stc_averages['right fist']

    # Visualize the difference
    print("Visualizing the difference between 'left fist' and 'right fist'...")
    brain_diff = stc_difference.plot(
        subject='fsaverage', subjects_dir=subjects_dir, initial_time=0.1,
        hemi='both', time_viewer=True, colormap='coolwarm',
        clim=dict(kind='value', lims=[-3, 0, 3]),
        title='Difference: Left Fist - Right Fist'
    )

    # Save the movie
    movie_output_path = os.path.join(output_dir, f'{base_filename}_left_fist_minus_right_fist.mp4')
    brain_diff.save_movie(movie_output_path, time_dilation=10, tmin=0.0, tmax=stc_difference.times[-1], framerate=20)
    print(f"Brain activity difference movie saved to {movie_output_path}")
    brain_diff.close()
else:
    print("Conditions 'left fist' and 'right fist' not found in data.")

# Example: Difference between 'both fists' and 'both feet'
if 'both fists' in stc_averages and 'both feet' in stc_averages:
    print("Computing difference between 'both fists' and 'both feet'...")
    stc_difference = stc_averages['both fists'] - stc_averages['both feet']

    # Visualize the difference
    print("Visualizing the difference between 'both fists' and 'both feet'...")
    brain_diff = stc_difference.plot(
        subject='fsaverage', subjects_dir=subjects_dir, initial_time=0.1,
        hemi='both', time_viewer=True, colormap='coolwarm',
        clim=dict(kind='value', lims=[-3, 0, 3]),
        title='Difference: Both Fists - Both Feet'
    )

    # Save the movie
    movie_output_path = os.path.join(output_dir, f'{base_filename}_both_fists_minus_both_feet.mp4')
    brain_diff.save_movie(movie_output_path, time_dilation=10, tmin=0.0, tmax=stc_difference.times[-1], framerate=20)
    print(f"Brain activity difference movie saved to {movie_output_path}")
    brain_diff.close()
else:
    print("Conditions 'both fists' and 'both feet' not found in data.")

# --- Optional: Visualize Each Condition Separately ---
print("Visualizing each condition separately...")
for condition in conditions:
    print(f"Visualizing condition: {condition}")
    brain = stc_averages[condition].plot(
        subject='fsaverage', subjects_dir=subjects_dir, initial_time=0.1,
        hemi='both', time_viewer=True,
        title=f'Source Estimate for {condition.capitalize()}'
    )

    # Save the movie
    movie_output_path = os.path.join(output_dir, f'{base_filename}_{condition}_source_estimate.mp4')
    brain.save_movie(movie_output_path, time_dilation=10, tmin=0.0, tmax=stc_averages[condition].times[-1], framerate=20)
    print(f"Brain activity movie for {condition} saved to {movie_output_path}")
    brain.close()

# End of script
print("Script completed successfully.")
