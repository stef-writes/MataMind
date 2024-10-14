import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import os

# Utility function to save figures
def save_figure(fig, filename, folder="figures"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig_path = os.path.join(folder, filename)
    fig.savefig(fig_path)
    print(f"Figure saved as {fig_path}")

# Load the epochs data dynamically
def load_epochs(directory):
    # Find the latest cleaned epochs file dynamically
    files = [f for f in os.listdir(directory) if f.endswith('_epochs-epo.fif')]
    if not files:
        raise FileNotFoundError("No cleaned epochs file found.")
    epochs_filepath = os.path.join(directory, max(files, key=os.path.getmtime))

    print(f"Loading epochs data from {epochs_filepath}...")
    epochs = mne.read_epochs(epochs_filepath, preload=True)

    # Rename channels to standardized names for compatibility with the montage
    rename_mapping = {name: name.replace('.', '').upper() for name in epochs.ch_names}
    epochs.rename_channels(rename_mapping)

    # Set a standard montage to ensure channel locations are present
    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage, on_missing='warn')
    print("Channel names have been standardized to match the montage.")
    return epochs

# Plot heatmap of channel activity
def plot_heatmap(epochs, save=False):
    print("Generating heatmap of EEG channel activity...")
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    avg_data = np.mean(data, axis=0)  # Average across epochs (n_channels, n_times)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(avg_data, cmap='viridis', cbar=True, xticklabels=20, yticklabels=epochs.ch_names, ax=ax)
    ax.set_xlabel('Time Points')
    ax.set_ylabel('Channels')
    ax.set_title('Heatmap of Channel Activity')
    if save:
        save_figure(fig, "heatmap_channel_activity.png")
    plt.show()

# Plot butterfly plot with GFP overlay
def plot_butterfly(epochs, save=False):
    print("Plotting butterfly plot of all channels...")
    evoked = epochs.average()

    # Plot the evoked response with colored lines
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(evoked.ch_names)))
    for idx, channel in enumerate(evoked.ch_names):
        ax.plot(evoked.times, evoked.data[idx], color=colors[idx], alpha=0.6, label=channel)

    # Calculate and plot the GFP (Global Field Power)
    gfp = np.sqrt(np.mean(evoked.data**2, axis=0))
    ax.plot(evoked.times, gfp, color='black', linewidth=2, label='GFP')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Butterfly Plot with GFP Overlay')
    ax.legend(loc='upper right', fontsize='small', ncol=2)
    ax.grid(True)
    if save:
        save_figure(fig, "butterfly_plot_with_gfp.png")
    plt.show()

# Plot topographic maps at specific time points, excluding overlapping electrodes
def plot_topomaps(evoked, times, save=False):
    print("Generating topographic maps at specific time points...")
    # Exclude electrodes with overlapping positions
    bad_channels = ['FCZ', 'CZ', 'CPZ', 'FP1', 'FPZ', 'FP2', 'AFZ', 'FZ', 'PZ', 'POZ', 'OZ', 'IZ']
    evoked.pick_channels([ch for ch in evoked.ch_names if ch not in bad_channels])

    fig = evoked.plot_topomap(times=times, ch_type='eeg', cmap='viridis', show=True)
    fig.suptitle('Topographic Maps', fontsize=14)
    if save:
        save_figure(fig, "topographic_maps.png")
    plt.show()

# Main function to execute the visualizations
def main():
    # Directory where the cleaned epochs files are saved (should match ICA script output)
    directory = '/Users/nooz/MetaViz/MataMind/scripts/'

    # Load epochs data dynamically
    epochs = load_epochs(directory)

    # Generate visualizations
    plot_heatmap(epochs, save=True)
    plot_butterfly(epochs, save=True)
    plot_topomaps(epochs.average(), times=[0.1, 0.2, 0.3], save=True)  # Example times in seconds

    print("Visualizations complete.")

if __name__ == "__main__":
    main()
