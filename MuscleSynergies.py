import numpy as np
import matplotlib.pyplot as plt
import wfdb

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

if __name__ == "__main__":
    # Load data
    records = ['data/dynamic_preprocess_sample60', 'data/dynamic_preprocess_sample66']
    emg_signal = [wfdb.rdrecord(record).p_signal for record in records]
    emg_signal = np.concatenate(emg_signal, axis=0)
    fs = wfdb.rdrecord(records[0]).fs

    time = np.arange(emg_signal.shape[0]) / fs 
    print(f'Sampling frequency: {fs}')
    print("emg_signal.shape: ", emg_signal.shape)

    #plot first channel of EMG signal
    # plot_emg_channel(emg_signal, time, channel_idx=1, save=False, file_name = 'emg_channel1.png')

    # Compute RMS of the EMG signal and remove noisy channels
    rms = compute_RMS(emg_signal)
    # plot_rms_values(rms, save=False, file_name = 'rms_values.png')

    nosisy_channels = np.where(rms > 0.2)[0]
    print(f'Channels with RMS > 0.2: {nosisy_channels + 1}')

    emg_signal_clean = np.delete(emg_signal, nosisy_channels, axis=1)
    print("original emg_signal.shape: ", emg_signal.shape)
    print("After removing noisy channels: ", emg_signal_clean.shape)


