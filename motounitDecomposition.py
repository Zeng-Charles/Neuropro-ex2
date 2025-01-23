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
        spike_train[indices, i] = 1
    
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

    # print("window_size_samples: ", window_size_samples)
    # print("num_windows: ", num_windows)

    firing_frequency = np.zeros((num_units, num_windows), dtype=float)

    for i in range(num_units):
        # print(f"calculating firing frequency for # MU{i+1}")
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


def plot_MU_spike_trian_with_force(ref_signal, time, spike_train, save=False):
    plt.figure(figsize=(20, 12))
    plt.suptitle('Motor Unit Spike Train', fontsize=14)

    for i in range(4):
        ax1 = plt.subplot(4, 1, i + 1)
        
        # left y axis -- spike train
        ax1.stem(time, spike_train[:, i], linefmt=plt.cm.viridis(i / 5), 
             markerfmt='.', basefmt=" ", label=f'MU# {i + 1} (Spike)')
        ax1.set_ylim(-0.1, 1.3)
        ax1.set_xlabel('Time (s)', fontsize=8)
        ax1.set_ylabel('Spike', fontsize=8, color=plt.cm.viridis(i / 5))
        ax1.tick_params(axis='y', labelcolor=plt.cm.viridis(i / 5))
        ax1.grid(True)
        
        # right y axis -- force signal
        ax2 = ax1.twinx()
        ax2.plot(time, ref_signal[0], label='Force Signal', color='red')
        ax2.set_ylabel('Force', fontsize=8, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # set title and legend
        ax1.set_title(f'MU# {i + 1}', fontsize=10)
        ax1.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig('spike_train.png')
    plt.show()

def plot_emg_signals_in1polt(emg_data, time, first_n_channels=10, save=False):
    plt.figure(figsize=(12, 6))
    for i in range(first_n_channels):
        plt.plot(time, emg_data[i], label=f'Channel {i + 1}', color=plt.cm.viridis(i / first_n_channels))
    plt.title('EMG Signal', fontsize=14)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.grid(True) 
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig('emg_signal_in1plot.png') 
    plt.show()

def plot_emg_channels(emg_data, time, first_n_channels=10, save=False):
    plt.figure(figsize=(10, 20))

    for i in range(first_n_channels):
        plt.subplot(first_n_channels, 1, i + 1)
        plt.plot(time, emg_data[i], label=f'Channel {i + 1}', color=plt.cm.viridis(i / first_n_channels))
        plt.title(f'Channel {i + 1}', fontsize=10)
        plt.xlabel('Time (s)', fontsize=8)
        plt.ylabel('Amplitude', fontsize=8)
        plt.grid(True)
        plt.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig('emg_signal.png') 
    plt.show()

def plot_firing_frequency_total(firing_frequency_total, time,  window_size=50, save=False):
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Firing Frequency Window Size {window_size}ms', fontsize=14)

    plt.plot(time[:firing_frequency_total.shape[0]], firing_frequency_total, label=f'Window Size {window_size} ms')

    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Firing Frequency (Hz)', fontsize=10)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=10)
    if save:
        plt.savefig(f'firing_frequency_{window_size}ms.png')
    plt.show()
   

def plot_cumulative_spike_train(time, CST, save=False):
    plt.figure()
    plt.suptitle('Cumulative Spike Train', fontsize=14)
    plt.plot(time, CST, label='CST')
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('CST', fontsize=10)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=10)
    if save:
        plt.savefig('CST.png')
    plt.show()

def plot_motor_unit_firing_frequency(time,index_motor_unit, firing_frequencies_unit, window_size_ms = 50, save=False):
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'#MU{index_motor_unit} Firing Frequency Window size {window_size_ms}ms', fontsize=14)
    plt.plot(time[ :firing_frequencies_unit.shape[1]], firing_frequencies_unit[index_motor_unit-1,:], label=f"#MU{index_motor_unit}, window size ms{window_size_ms}")
    plt.xlabel("Time (s)", fontsize=10)
    plt.ylabel("Firing Frequency (Hz)", fontsize=10)
    plt.grid(True)
    plt.legend()
    if save:
        plt.savefig(f'firing_frequency_unit #MU{index_motor_unit} {window_size_ms}ms.png')
    plt.show()

if __name__ == '__main__':
    # Load the .mat file
    data = scipy.io.loadmat('data/GM_10.mat')

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
    plot_emg_channels(emg_data, time, first_n_channels=10, save=False)

    # plot the emg signal in same plot
    plot_emg_signals_in1polt(emg_data, time, first_n_channels=10, save=False)

    ###################################################### Decomposition ####################################################################
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
    print("lenth of firing_indices 1: ", len(firing_indices[0]))
    print("lenth of firing_indices 2: ", len(firing_indices[1]))


    # Convert motor unit pulses to a spike train
    spike_train = convert_to_spike_train(decomp, firing_indices, emg_data.shape[1], threshold=0)
    print("spike_train shape: ", spike_train.shape)

    # Plot the motor unit spike train and the force signal
    plot_MU_spike_trian_with_force(ref_signal, time, spike_train)  

    ###################################################### cumlative spike train ####################################################################

    #cumulative spike train
    cumulative_spike_train = np.zeros(spike_train.shape)

    for i in range(num_units):
        cumulative_spike_train[:, i] = np.cumsum(spike_train[:, i])

    print("cumulative_spike_train shape: ", cumulative_spike_train.shape)
    CST = np.sum(cumulative_spike_train, axis=1)
    print("CST shape: ", CST.shape)

    # Plot CST
    plot_cumulative_spike_train(time, CST, save=True)

    # Calculate the firing frequency total
    window_sizes = [50, 100, 200]
    firing_frequencies_total = [calculate_firing_frequency_total(spike_train, fsamp, window_size_ms = ws) for ws in window_sizes]
    #plot the firing frequency total
    for i, ws in enumerate(window_sizes):
        plot_firing_frequency_total(firing_frequencies_total[i], time, window_size=ws, save=False)

    # Calculate the firing frequency of the each motor units
    firing_frequencies_unit = calculate_firing_frequency_signal(spike_train, fsamp, window_size_ms = 50)
    print("firing_frequencies_unit shape: ", firing_frequencies_unit.shape)
    # Plot the firing frequency of the motor unit
    index_motor_unit = 1
    plot_motor_unit_firing_frequency(time, index_motor_unit, firing_frequencies_unit, window_size_ms=50, save=False)



##########################################################################part 3####################################################################

# Spike-Triggers Averaging
