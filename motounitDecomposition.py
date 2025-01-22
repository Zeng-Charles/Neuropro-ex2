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

# Load the .mat file
data = scipy.io.loadmat('/Users/a1/Desktop/EX2/Experimental_data_Raw/GL_10.mat')

# Access the variables in the .mat file
print(data.keys())

# Extract the variables
SIG = data['SIG']
ref_signal = data['ref_signal']
fsamp = data['fsamp']

print("SIG shape: ", SIG.shape)
print("ref_signal shape: ", ref_signal.shape)
print( "fsamp shape: ", fsamp.shape)

# Concatenate all non-empty channels of the EMG signal
emg_data = np.vstack([channel for row in SIG for channel in row if channel.size > 0])
print("emg_data shape: ", emg_data.shape)

time = np.arange(emg_data.shape[1]) / fsamp[0, 0]  # Convert samples to time in seconds

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
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.plot(time, emg_data[i], label=f'Channel {i + 1}', color=plt.cm.viridis(i / 10))
plt.title('EMG Signal', fontsize=14)
plt.xlabel('Time (s)', fontsize=10)
plt.ylabel('Amplitude', fontsize=10)
plt.grid(True) 
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()

# plt.savefig('emg_signal_1plot.png') 
# plt.show()

#Decompose the EMG signal
a = input("decomposition or load file (0/1): ")

if a == '0':
    output = decomposition(
        emg_data,
        discard=5,
        R=16,
        M=64,
        bandpass=True,
        lowcut=10,
        highcut=900,
        fs=2048,
        order=6,
        Tolx=10e-4,
        contrast_fun=skew,
        ortho_fun=gram_schmidt,
        max_iter_sep=10,
        l=31,
        sil_pnr=True,
        thresh=0.9,
        max_iter_ref=10,
        random_seed=None,
        verbose=False
    )

    # Save the output
    decomp_GL_10 = output 
    decomp_GL_10_pkl = open('decomp_GL_10_pkl.obj', 'wb') 
    pickle.dump(decomp_GL_10, decomp_GL_10_pkl)

elif a == '1':

    # Load the output
    with open('decomp_GL_10_pkl.obj', 'rb') as f: output = pickle.load(f)
    decomp_GL_10 = output
    print("output dictionary keys: ", output.keys())

    
# Extract the decomposition 
decomp = decomp_GL_10['B']
num_units = decomp.shape[1]
firing_indices = decomp_GL_10['MUPulses']
print("decomp shape: ", decomp.shape)
print("num_units: ", num_units)
print("firing_indices shape: ", firing_indices.shape)

# Convert motor unit pulses to a spike train
spike_train = convert_to_spike_train(decomp, firing_indices, emg_data.shape[1], threshold=0)
print("spike_train shape: ", spike_train.shape)

# Plot the motor unit spike train and the force signal
plt.figure(figsize=(12, 14))
plt.suptitle('Motor Unit Spike Train', fontsize=14)

for i in range(4):
    ax1 = plt.subplot(4, 1, i + 1)
    
    # left y axis -- spike train
    ax1.set_ylim(-0.1, 1.3)
    ax1.step(time, spike_train[:, i], label=f'MU# {i + 1} (Spike)', 
             color=plt.cm.viridis(i / 5), where='post')
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
#plot the force signal
# plt.subplot(5, 1, 5)
# plt.plot(time, ref_signal[0], label='force signal', color=plt.cm.viridis(5 / 5))
# plt.title('Force Signal', fontsize=10)
# plt.xlabel('Time (s)', fontsize=8)
# plt.ylabel('Amplitude', fontsize=8)
# plt.grid(True)
# plt.legend(loc='upper right', fontsize=8)
# plt.tight_layout()

plt.savefig('spike_train.png') # Save the plot as an image
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

# Plot CST
plt.figure()
plt.plot(time, CST, label='CST', color='blue')
plt.show()