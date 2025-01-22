# Neuropro Excrise 2--Chaoyu Zeng
## Introduction

This project is part of Neuropro Exercise 2, which involves the decomposition of EMG signals to analyze motor unit activity. The provided Python script `motounitdecomposition.py` performs the following tasks:

1. Loads EMG data from a `.mat` file.
2. Preprocesses and visualizes the EMG data.
3. Decomposes the EMG signal into motor unit action potentials (MUAPs).
4. Converts the motor unit pulses into a spike train.
5. Visualizes the spike train.

## Features

- **Loading EMG Data**: The script loads EMG data from a specified `.mat` file and extracts relevant variables.
- **Data Visualization**: It visualizes the EMG signals from different channels for better understanding and analysis.
- **EMG Signal Decomposition**: The script decomposes the EMG signal using the `emgdecompy` library, which includes various preprocessing and decomposition techniques.
- **Spike Train Conversion**: Converts the decomposed motor unit pulses into a spike train for further analysis.
- **Spike Train Visualization**: Visualizes the spike train to observe the firing patterns of motor units.

## Author
Chaoyu Zeng