import numpy as np
import matplotlib.pyplot as plt
import wfdb
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF

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
    # scaler = StandardScaler()
    # data_std = scaler.fit_transform(data)

    pca = PCA(num_components)
    data_pca = pca.fit_transform(data)

    data_reconstructed = pca.inverse_transform(data_pca)

    return data_pca, data_reconstructed

def Non_negative_Matrix_Factorization(data, num_components=4):
    """
    Perform Non-negative Matrix Factorization (NMF) on the EMG signal

    Parameters
    ----------
    data : 2d numpy array in shape (n_samples, n_channels)  
        EMG signal
    num_components : int
        Number of components to extract

    Returns
    -------
    W : 2d numpy array in shape (n_samples, num_components)
        weights of respective channels
    H : 2d numpy array in shape (num_components, n_channels)
        Dynamic changes in synergy
    data_reconstructed : 2d numpy array in shape (n_samples, n_channels)
        Reconstructed data after NMF
    """
    # scaler = MinMaxScaler()
    # data_std = scaler.fit_transform(data)
    min_value = np.min(data)
    if min_value < 0:
        data_shifted = data - min_value
    else:
        data_shifted = data

    nmf = NMF(num_components, max_iter=100000, init='random', random_state=42)
    W = nmf.fit_transform(data_shifted)

    H = nmf.components_

    data_reconstructed = np.dot(W, H) + min_value

    return W, H, data_reconstructed

def Run_NMF_different_components(data, components=[]):
    """
    Perform Non-negative Matrix Factorization (NMF) on the EMG signal with different number of components
    
    Parameters
    ----------
    data : 2d numpy array in shape (n_samples, n_channels)  
        EMG signal
    components : list
        List of number of components to extract

    Returns
    -------
    W : list of 2d numpy array in shape (n_samples, num_components)
        weights of respective channels
    H : list of 2d numpy array in shape (num_components, n_channels)
        Dynamic changes in synergy
    reconstructions : list of 2d numpy array in shape (n_samples, n_channels)    
        Reconstructed data after NMF
    r2_NMF : 2d numpy array in shape (len(components), n_channels)
        R2 values for each channel with different number of components
    r2_NMF_mean : 1d numpy array in shape (len(components))
        Mean R2 values for each number of components
    """
    
    r2_NMF = np.zeros((len(components), emg_signal.shape[1]))
    r2_NMF_mean = np.zeros(len(components))
    W, H, reconstructions = [], [], []

    for i in components:
        print (f'Running NMF with {i} components')
        W_i, H_i, reconstructed = Non_negative_Matrix_Factorization(data, i)
        W.append(W_i)
        H.append(H_i)
        reconstructions.append(reconstructed)
        r2_NMF[i - 1, :] = compute_r2(emg_signal, reconstructed)
        r2_NMF_mean[i - 1] = np.mean(r2_NMF[i - 1, :])
        print(f'R2_NMF mean for {i} components: {r2_NMF_mean[i - 1]:.4f}')

    return W, H, reconstructions, r2_NMF, r2_NMF_mean

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

def plot_origin_vs_reconstructed(time, original, reconstructed_pca = None,  reconstructed_nmf = None, save=False, file_name = None):
    plt.figure(figsize=(10, 6))
    plt.suptitle("Original vs Reconstructed EMG signal", fontsize=16)
    plt.plot(time[0:500], original[0:500], label = "Original EMG signal", color = '#1f77b4')
    if reconstructed_pca is not None:
        plt.plot(time[0:500], reconstructed_pca[0:500], label = "PCA Reconstructed EMG signal", color = '#ff7f0e')
    if reconstructed_nmf is not None:
        plt.plot(time[0:500], reconstructed_nmf[0:500], label = "NMF Reconstructed EMG signal", color = '#2baf6a')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    
    if save:
        plt.savefig(file_name)
    plt.show()

def plot_W(W, save=False, file_name = None):
    plt.figure(figsize=(16, 9))
    plt.suptitle("Weight of channels", fontsize=16)
    for i in range (W.shape[1]):
        plt.subplot(W.shape[1],1,i+1)
        plt.plot(W[:, i], label = f"Component #{i+1}", color = plt.cm.viridis(i/W.shape[1]))
        plt.title(f"Component {i+1}")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')

    plt.tight_layout()
    if save:
        plt.savefig(file_name)
    plt.show()

def plot_H(H, save=False, file_name = None):
    plt.figure(figsize=(16, 9))
    plt.suptitle("Dynamic Changes in Synergy", fontsize=16)
    for i in range (H.shape[0]):
        plt.subplot(H.shape[0],1,i+1)
        plt.plot(H[i, :], label = f"Component #{i+1}", color = plt.cm.viridis(i/H.shape[0]))
        plt.title(f"Component {i+1}")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')

    plt.tight_layout()
    if save:
        plt.savefig(file_name)
    plt.show()

def plot_r2_nmf_values(r2_NMF_mean, components, save=False, file_name = None):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(components, r2_NMF_mean)
    plt.title('R2 values of NMF with different number of components')
    plt.xlabel('Number of components')
    plt.ylabel('R2 value')
    plt.xticks(components)
    plt.grid(axis='y')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', ha='center', va='bottom')

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
    emg_pca, emg_reconstructed_PCA= Principle_Component_Analysis(emg_signal, num_components)
    # plot components of EMG signal
    # plot_emg_components(time, num_components, emg_pca, save = False, file_name = 'emg_components.png')

    r2_PCA = compute_r2(emg_signal, emg_reconstructed_PCA)
    r2_PCA_mean = np.mean(r2_PCA)
    print(f'R2_PCA mean: {r2_PCA_mean}')
    #plot first channel of original and reconstructed EMG signal
    # plot_origin_vs_reconstructed(time, emg_signal[:, 0], emg_reconstructed_PCA[:, 0], save=False, file_name = 'origin_vs_PCA_reconstructed.png')

    #Non-negative Matrix Factorization
    num_components = 4
    W, H, emg_reconstructed_NMF = Non_negative_Matrix_Factorization(emg_signal, num_components)
    # plot components of EMG signal
    # plot_W(W, save = False, file_name = 'W.png')
    # plot_H(H, save = False, file_name = 'H.png')

    #plot first channel of original and reconstructed EMG signal
    # plot_origin_vs_reconstructed(time, emg_signal[:, 0], reconstructed_pca= None, reconstructed_nmf = emg_reconstructed_NMF[:, 0],save=False, file_name = 'origin_vs_NMF_reconstructed.png')
    r2_NMF = compute_r2(emg_signal, emg_reconstructed_NMF)
    r2_NMF_mean = np.mean(r2_NMF)
    print(f'R2_NMF mean: {r2_NMF_mean}')

    #plot PCA and NMF comparison
    # plot_origin_vs_reconstructed(time, emg_signal[:, 0], emg_reconstructed_PCA[:, 0], emg_reconstructed_NMF[:, 0], save=False, file_name = 'origin_vs_PCA_NMF_reconstructed.png')

    #Use NMF to extract muscle synergies 1 to 15 components
    components = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    W, H, reconstructions, r2_NMF, r2_NMF_mean = Run_NMF_different_components(emg_signal, components)

    #plot R2 values
    plot_r2_nmf_values(r2_NMF_mean, components, save=False, file_name = 'r2_NMF_values.png')



