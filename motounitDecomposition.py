import scipy
import pickle
import altair as alt

import numpy as np
import matplotlib.pyplot as plt

from emgdecompy.decomposition import *
from emgdecompy.contrast import *
from emgdecompy.viz import *
from emgdecompy.preprocessing import *

def convert_to_spike_train(data, firing_indices, emg_len, threshold = 0):
    """
    Convert motor unit pulses to a spike train
    Parameters
    ----------
    data : ndarray
        motor unit pulses.
    emg_len : int
        length of the EMG signal.
    firing_indices : ndarray
        indices of the motor unit pulses.
    threshold (optional): float
        threshold for the motor unit pulses.
        
    Returns
    -------
    spike_train : ndarray
        The spike train.
    """
    num_units = data.shape[1]
    
    spike_train = np.zeros((emg_len, num_units))

    for i in range(num_units):
        indices = firing_indices[i]
        for p, time_idx in enumerate(indices):
            if 0 <= time_idx < emg_len:
                if data[p, i] > threshold:
                    spike_train[time_idx, i] = 1
    
    return spike_train

def calculate_firing_frequency_signal(spike_train, fs, window_size_ms = 50):
    """
    Calculate the firing frequency of the motor units
    Parameters:
    ----------
    spike_train : ndarray, shape (n_samples, n_units)
        The spike train.
    fs : int
        Sampling frequency.
    window_size_ms : int(ms)

    Returns
    -------
    firing_frequency : ndarray
        The firing frequency of the motor units.
    """
    window_size_samples = float(fs) * float(window_size_ms) / 1000.0
    window_size_samples = int(window_size_samples)
    num_samples, num_units = spike_train.shape
    num_windows = num_samples - window_size_samples + 1

    print("window_size_samples: ", window_size_samples)
    print("num_windows: ", num_windows)

    firing_frequency = np.zeros((num_units, num_windows), dtype=float)

    for i in range(num_units):
        print(f"calculating firing frequency for # MU{i+1}")
        conv_result = np.convolve(spike_train[:, i], np.ones(window_size_samples), mode='valid')
        firing_frequency[i, :] = conv_result / (window_size_ms * 0.001) 

    return firing_frequency

def calculate_firing_frequency_total(spike_train, fs, window_size_ms = 50):
    """
    Calculate the firing frequency of the motor units
    Parameters:
    ----------
    spike_train : ndarray, shape (n_samples, n_units)
        The spike train.
    fs : int
        Sampling frequency.
    window_size : int(ms)

    Returns
    -------
    firing_frequency : ndarray
        The firing frequency of the motor units.
    """
    window_size_samples = float(fs) * float(window_size_ms) / 1000.0
    window_size_samples = int(window_size_samples)
    firing_frequency = np.convolve(np.sum(spike_train, axis=1), np.ones(window_size_samples), mode='valid') / window_size_ms * 1000

    return firing_frequency

def calculate_spike_triggered_average(spike_train, fs, window_size_ms = 50):
    """
    Calculate the spike-triggered average of the EMG signal
    Parameters:
    ----------
    spike_train : ndarray, shape (n_samples, n_units)
        The spike train.
    fs : int
        Sampling frequency.
    window_size : int(ms)

    Returns
    -------
    sta : ndarray
        The spike-triggered average of the EMG signal.
    """
    window_size_samples = float(fs) * float(window_size_ms) / 1000.0
    window_size_samples = int(window_size_samples)
    num_units = spike_train.shape[1]
    sta = np.zeros((window_size_samples, num_units))

    for i in range(num_units):
        indices = np.where(spike_train[:, i] == 1)[0]
        for idx in indices:
            if idx + window_size_samples < len(spike_train):
                sta += spike_train[idx:idx+window_size_samples, :]

    sta /= len(indices)

    return sta


if __name__ == '__main__':
    # Load the .mat file
    data = scipy.io.loadmat('Experimental_data_Raw/GM_10.mat')

    # Access the variables in the .mat file
    print(data.keys())

    # Extract the variables
    SIG = data['SIG']
    ref_signal = data['ref_signal']
    fsamp = data['fsamp']
    fsamp = fsamp[0,0]

    print("SIG shape: ", SIG.shape)
    print("ref_signal shape: ", ref_signal.shape)
    print("fsamp: ", fsamp)

    # Concatenate all non-empty channels of the EMG signal
    emg_data = np.vstack([channel for row in SIG for channel in row if channel.size > 0])
    print("emg_data shape: ", emg_data.shape)

    time = np.arange(emg_data.shape[1]) / fsamp # Convert samples to time in seconds

    # Plot the EMG signal
    # plt.figure(figsize=(10, 20))

    # for i in range(10):
    #     plt.subplot(10, 1, i + 1)
    #     plt.plot(time, emg_data[i], label=f'Channel {i + 1}', color=plt.cm.viridis(i / 10))
    #     plt.title(f'Channel {i + 1}', fontsize=10)
    #     plt.xlabel('Time (s)', fontsize=8)
    #     plt.ylabel('Amplitude', fontsize=8)
    #     plt.grid(True)
    #     plt.legend(loc='upper right', fontsize=8)

    # plt.tight_layout()
    # plt.show()

    #plot the emg signal in same plot
    # plt.figure(figsize=(12, 6))
    # for i in range(10):
    #     plt.plot(time, emg_data[i], label=f'Channel {i + 1}', color=plt.cm.viridis(i / 10))
    # plt.title('EMG Signal', fontsize=14)
    # plt.xlabel('Time (s)', fontsize=10)
    # plt.ylabel('Amplitude', fontsize=10)
    # plt.grid(True) 
    # plt.legend(loc='upper right', fontsize=10)
    # plt.tight_layout()

    # plt.savefig('emg_signal_1plot.png') 
    # plt.show()

    #Decompose the EMG signal
    a = input("decomposition or load file (0/1): ")

    if a == '0':
        output = decomposition(emg_data, fs=fsamp)

        # Save the output
        decomp_GM_10 = output 
        decomp_GM_10_pkl = open('decomp_GM_10_pkl.obj', 'wb') 
        pickle.dump(decomp_GM_10, decomp_GM_10_pkl)

    elif a == '1':

        # Load the output
        with open('decomp_GM_10_pkl.obj', 'rb') as f: output = pickle.load(f)
        decomp_GM_10 = output
        print("output dictionary keys: ", output.keys())

        
    # Extract the decomposition 
    decomp = decomp_GM_10['B']
    num_units = decomp.shape[1]
    firing_indices = decomp_GM_10['MUPulses']
    print("decomp shape: ", decomp.shape)
    print("num_units: ", num_units)
    print("firing_indices shape: ", firing_indices.shape)

    # Convert motor unit pulses to a spike train
    spike_train = convert_to_spike_train(decomp, firing_indices, emg_data.shape[1], threshold=0)
    print("spike_train shape: ", spike_train.shape)

    # Plot the motor unit spike train and the force signal
    # plt.figure(figsize=(12, 14))
    # plt.suptitle('Motor Unit Spike Train', fontsize=14)

    # for i in range(4):
    #     ax1 = plt.subplot(4, 1, i + 1)
        
    #     # left y axis -- spike train
    #     ax1.set_ylim(-0.1, 1.3)
    #     ax1.step(time, spike_train[:, i], label=f'MU# {i + 1} (Spike)', 
    #              color=plt.cm.viridis(i / 5), where='post')
    #     ax1.set_xlabel('Time (s)', fontsize=8)
    #     ax1.set_ylabel('Spike', fontsize=8, color=plt.cm.viridis(i / 5))
    #     ax1.tick_params(axis='y', labelcolor=plt.cm.viridis(i / 5))
    #     ax1.grid(True)
        
    #     # right y axis -- force signal
    #     ax2 = ax1.twinx()
    #     ax2.plot(time, ref_signal[0], label='Force Signal', color='red')
    #     ax2.set_ylabel('Force', fontsize=8, color='red')
    #     ax2.tick_params(axis='y', labelcolor='red')
        
    #     # set title and legend
    #     ax1.set_title(f'MU# {i + 1}', fontsize=10)
    #     ax1.legend(loc='upper left', fontsize=8)
    #     ax2.legend(loc='upper right', fontsize=8)

    # plt.tight_layout()
    #plot the force signal
    # plt.subplot(5, 1, 5)
    # plt.plot(time, ref_signal[0], label='force signal', color=plt.cm.viridis(5 / 5))
    # plt.title('Force Signal', fontsize=10)
    # plt.xlabel('Time (s)', fontsize=8)
    # plt.ylabel('Amplitude', fontsize=8)
    # plt.grid(True)
    # plt.legend(loc='upper right', fontsize=8)
    # plt.tight_layout()

    # plt.savefig('spike_train.png') # Save the plot as an image
    # plt.show()  

    ###################################################### part 2 ####################################################################

    #cumulative spike train

    cumulative_spike_train = np.zeros(spike_train.shape)

    for i in range(num_units):
        cumulative_spike_train[:, i] = np.cumsum(spike_train[:, i])

    print("cumulative_spike_train shape: ", cumulative_spike_train.shape)

    # Plot the cumulative spike train
    # plt.figure()
    # plt.plot(time, cumulative_spike_train[: , 0], label='MU# 1', color='blue')
    # plt.show()

    CST = np.sum(cumulative_spike_train, axis=1)
    print("CST shape: ", CST.shape)

    # Plot CST
    # plt.figure()
    # plt.suptitle('Cumulative Spike Train', fontsize=14)
    # plt.plot(time, CST, label='CST')
    # plt.xlabel('Time (s)', fontsize=10)
    # plt.ylabel('CST', fontsize=10)
    # plt.grid(True)
    # plt.legend(loc='upper left', fontsize=10)
    # plt.savefig('CST.png') # Save the plot as an image
    # plt.show()

    # Calculate the firing frequency of the motor units
    # window_sizes = [50, 100, 200]  # 50 ms, 100 ms, 200 ms
    # firing_frequencies_total = {}

    # plt.figure(figsize=(16, 12))
    # for i, ws in enumerate(window_sizes):
    #     firing_frequencies_total[ws] = calculate_firing_frequency_total(spike_train, fsamp, window_size_ms=ws)

    #     # Plot results
    #     plt.subplot(3,1,i+1)
    #     plt.suptitle('Total Firing Frequency', fontsize=14)
    #     plt.plot(time[:len(firing_frequencies_total[ws])], firing_frequencies_total[ws], label=f"Window Size: {int(ws)} ms",color=plt.cm.viridis(i / 3))
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Firing Frequency (Hz)")
    #     plt.title(f"Firing Frequency ({int(ws)} ms Window)")
    #     plt.legend()
    #     plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('firing_frequency_total.png')

    window_size_ms = 50
    index_motor_unit = 0
    firing_frequencies_unit = calculate_firing_frequency_signal(spike_train, fsamp, window_size_ms)
    print("firing_frequencies_unit shape: ", firing_frequencies_unit.shape)
    # Plot results
    # plt.figure(figsize=(12, 6))
    # plt.suptitle(f'#MU{index_motor_unit + 1} Firing Frequency', fontsize=14)
    # plt.plot(time[ :firing_frequencies_unit.shape[1]], firing_frequencies_unit[index_motor_unit,:], label=f"#MU{index_motor_unit+1}, window size ms{window_size_ms}")
    # plt.xlabel("Time (s)", fontsize=10)
    # plt.ylabel("Firing Frequency (Hz)", fontsize=10)
    # plt.grid(True)
    # plt.legend()
    # plt.savefig('firing_frequency_unit.png')
    # plt.show()



##########################################################################part 3####################################################################

# Spike-Triggers Averaging
