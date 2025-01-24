import numpy as np
import matplotlib.pyplot as plt
import wfdb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_emg_channel(emg_signal, time, channel_idx=0, save=False, file_name = None):
    """
    Plot the EMG signal of a specific channel

    Parameters
    ----------
    emg_signal : 2d numpy array in shape (n_samples, n_channels)  
        EMG signal
    time : 1d numpy array
        Time vector
    channel_idx : int
        Index of the channel to plot

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 5))
    plt.suptitle(f'Channel {channel_idx} of EMG Signal ')
    plt.plot(time, emg_signal[:, channel_idx - 1], label=f'Channel {channel_idx} of EMG signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    if save:
        plt.savefig(file_name)
    plt.show()

def compute_RMS(emg_signal):
    """
    Compute the root mean square (RMS) of the EMG signal

    Parameters
    ----------
    emg_signal : 2d numpy array in shape (n_samples, n_channels)  
        EMG signal

    Returns
    -------
    rms : 1d numpy array in shape (n_channels)
        RMS of the EMG signal
    """
    num_samples, num_channels = emg_signal.shape
    rms = np.zeros(num_channels)
    for channel in range(num_channels):
        rms[channel] = np.sqrt(np.mean(emg_signal[:, channel] ** 2))
    
    return rms

def plot_rms_values(rms, save=False, file_name = None):
    """
    Plot the RMS values of the EMG signal

    Parameters
    ----------
    rms : 1d numpy array in shape (n_channels)
        RMS of the EMG signal

    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(rms)), rms)
    plt.axhline(0.2, color='r', linestyle='--', label='RMS > 0.2')
    plt.title('RMS Values of Channels')
    plt.xlabel('Channel Index')
    plt.ylabel('RMS Value')
    plt.legend()
    if save:
        plt.savefig(file_name)
    plt.show()

def Principle_Component_Analysis(data, num_components=4):
    """
    Perform Principle Component Analysis (PCA) on the EMG signal

    Parameters
    ----------
    data : 2d numpy array in shape (n_samples, n_channels)  
        EMG signal
    num_components : int
        Number of components to extract

    Returns
    -------
    data_pca : 2d numpy array in shape (n_samples, num_components)
        Data after PCA
    data_reconstructed : 2d numpy array in shape (n_samples, n_channels)
        Reconstructed data after PCA
    """
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)

    pca = PCA(num_components)
    data_pca = pca.fit_transform(data_std)

    data_reconstructed = pca.inverse_transform(data_pca)

    return data_pca, data_reconstructed

def compute_r2(original, reconstructed):
    correlation_matrix = np.corrcoef(original.T, reconstructed.T)
    correlation_coefficients = np.diag(correlation_matrix[:original.shape[1], original.shape[1]:])
    r2 = correlation_coefficients ** 2
    return r2

def plot_emg_components(time, num_components, pca, save=False, file_name = None):
    plt.figure(figsize=(16, 9))
    plt.suptitle("Extracted Components of EMG signal", fontsize=16)
    for i in range (num_components):
        plt.subplot(num_components,1,i+1)
        plt.plot(time, pca[:, i], label = f"Components #{i+1}", color = plt.cm.viridis(i/num_components))
        plt.title(f"Component {i+1}")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')

    plt.tight_layout()
    if save:
        plt.savefig(file_name)
    plt.show()

def plot_origin_vs_reconstructed(time, original, reconstructed, save=False, file_name = None):
    plt.figure(figsize=(10, 6))
    plt.suptitle("Original vs Reconstructed EMG signal", fontsize=16)
    plt.plot(time[0:500], original[0:500], label = "Original EMG signal", color = 'b')
    plt.plot(time[0:500], reconstructed[0:500], label = "Reconstructed EMG signal", color = 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    
    if save:
        plt.savefig(file_name)
    plt.show()

if __name__ == "__main__":
    # Load data
    records = ['data/dynamic_preprocess_sample60', 'data/dynamic_preprocess_sample66']
    data = [wfdb.rdrecord(record).p_signal for record in records]
    emg_signal_raw = np.concatenate(data, axis=0)
    fs = wfdb.rdrecord(records[0]).fs

    time = np.arange(emg_signal_raw.shape[0]) / fs 
    print(f'Sampling frequency: {fs}')
    print("emg_signal.shape: ", emg_signal_raw.shape)

    #plot first channel of EMG signal
    # plot_emg_channel(emg_signal, time, channel_idx=1, save=False, file_name = 'emg_channel1.png')

    # Compute RMS of the EMG signal and remove noisy channels
    rms = compute_RMS(emg_signal_raw)
    # plot_rms_values(rms, save=False, file_name = 'rms_values.png')

    nosisy_channels = np.where(rms > 0.2)[0]
    print(f'Channels with RMS > 0.2: {nosisy_channels + 1}')

    emg_signal = np.delete(emg_signal_raw, nosisy_channels, axis=1)
    print("original emg_signal.shape: ", emg_signal_raw.shape)
    print("After removing noisy channels: ", emg_signal.shape)

    #Principle Component Analysis
    num_components = 4
    emg_pca, emg_reconstructed= Principle_Component_Analysis(emg_signal, num_components)
    # plot components of EMG signal
    # plot_emg_components(time, num_components, emg_pca, save = False, file_name = 'emg_components.png')

    r2 = compute_r2(emg_signal, emg_reconstructed)
    print("R2.shape: ", r2.shape)
    #plot first channel of original and reconstructed EMG signal
    plot_origin_vs_reconstructed(time, emg_signal_raw[:, 0], emg_reconstructed[:, 0], save=False, file_name = 'origin_vs_reconstructed.png')

    




