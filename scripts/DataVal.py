import os
import sys
from config import EEG_FILE_PATH

# Set up the directory where the data is expected to be
data_directory = os.path.dirname(os.path.dirname(EEG_FILE_PATH))

# Function to validate if data files are available and organized correctly
def validate_data_directory(data_directory):
    print(f"Validating data in directory: {data_directory}...")
    
    # Check if the directory exists
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Data directory {data_directory} does not exist. Please check the path.")
    
    # Iterate through subdirectories to find EDF files
    missing_files = []
    for subject_id in range(1, 110):  # Updated range to cover all subjects (S001 to S109)
        subfolder = f"S{subject_id:03}"
        subject_path = os.path.join(data_directory, subfolder)
        if not os.path.exists(subject_path):
            missing_files.append(subfolder)
        else:
            edf_files = [f for f in os.listdir(subject_path) if f.endswith(".edf")]
            if len(edf_files) == 0:
                missing_files.append(subfolder)

    if missing_files:
        raise FileNotFoundError(f"Missing EDF files in subject folders: {missing_files}. Please check the data.")
    
    print("Data validation complete. All expected files are present.")

# Main function to run the data validation if executed as a script
if __name__ == "__main__":
    try:
        validate_data_directory(data_directory)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("Data is ready for preprocessing.")