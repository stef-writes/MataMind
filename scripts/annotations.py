import numpy as np
import mne
from config import EEG_FILE_PATH

# Load the raw EEG data using MNE
print("Loading raw EEG data using MNE...")
raw = mne.io.read_raw_edf(EEG_FILE_PATH, preload=True)

# Extract annotations from the raw data
print("Extracting annotations from the raw data...")
annotations = raw.annotations

# Print the annotations
print("Annotation onsets:", annotations.onset)
print("Annotation durations:", annotations.duration)
print("Annotation descriptions:", annotations.description)

# Optional: Save annotations to a text file if needed
annotations_file = 'S001R03_annotations.txt'
print(f"Saving annotations to {annotations_file}...")
with open(annotations_file, 'w') as f:
    for i in range(len(annotations.onset)):
        f.write(f"Onset: {annotations.onset[i]}, Duration: {annotations.duration[i]}, Description: {annotations.description[i]}\n")

# Create MNE annotations from the extracted data
print("Creating MNE annotations from the extracted data...")
onset = annotations.onset
duration = annotations.duration
description = annotations.description

# Create and set annotations in the raw object
new_annotations = mne.Annotations(onset=onset, duration=duration, description=description)
raw.set_annotations(new_annotations)

# Save the annotated raw data
annotated_raw_file = 'filtered_subject1_raw_with_annotations_raw.fif'
print(f"Saving annotated raw data to {annotated_raw_file}...")
raw.save(annotated_raw_file, overwrite=True)

print("Annotation extraction and saving complete.")
