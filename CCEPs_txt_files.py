# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 09:49:19 2025

@author: msedo
"""

import numpy as np
import re
import matplotlib.pyplot as plt


def load_inomed_eeg(filepath, start_time=None, end_time=None):
    """
    Load inomed EEG data and optionally filter by time range.
    
    Parameters:
    -----------
    filepath :  str
        Path to the inomed ASCII txt file
    start_time : str, optional
        Start time in HH:MM:SS format (e.g., "13:00:00")
    end_time : str, optional
        End time in HH:MM:SS format (e.g., "14:00:00")
    
    Returns:
    --------
    data : numpy.ndarray
        EEG data (channels x measurements)
    times : list
        Measurement times as strings
    sampling_rate : int
        Sampling rate in Hz
    """
    # Read file
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()
    
    # Extract sampling rate
    rate_match = re.search(r'Sampling rate:\s*(\d+)\s*Hz', content)
    sampling_rate = int(rate_match.group(1)) if rate_match else None
    
    # Extract measure times
    times_match = re.search(r'Measure times:(.*?)Data:', content, re.DOTALL)
    times = re.findall(r'\d{1,2}:\d{2}:\d{2}', times_match.group(1))
    
    # Extract data
    data_match = re.search(r'Data:\s*\n(.*?)$', content, re.DOTALL)
    data_lines = data_match. group(1).strip().split('\n')
    
    channels = []
    for line in data_lines: 
        line = re.sub(r'^\s*\d+\s*:', '', line.strip())
        values = [float(v) for v in re.findall(r'-?\d+\. ?\d*', line)]
        if values:
            channels.append(values)
    
    data = np.array(channels)
    
    # Filter by time range if provided
    if start_time and end_time: 
        indices = [i for i, t in enumerate(times) if start_time <= t <= end_time]
        data = data[:, indices]
        times = [times[i] for i in indices]
    
    return data, times, sampling_rate


def plot_cascade(data, times, num_signals=10, spacing=5):
    """
    Plot EEG signals in cascade form. 
    
    Parameters:
    -----------
    data : numpy.ndarray
        EEG data (channels x measurements)
    times : list
        Measurement times as strings
    num_signals : int
        Number of signals to plot (default: 10)
    spacing : float
        Vertical spacing between signals (default: 5)
    """
    # Limit to requested number of signals
    num_signals = min(num_signals, data.shape[1])
    
    # Create figure
    plt. figure(figsize=(14, 10))
    
    # Get number of channels
    num_channels = data. shape[0]
    
    # Plot each signal with vertical offset
    for i in range(num_signals):
        offset = (num_signals - i - 1) * spacing
        
        for ch in range(num_channels):
            plt.plot(data[ch, i] + offset, 
                    linewidth=1.5, 
                    label=f'Ch{ch+1}' if i == 0 else '')
        
        # Add time label on the left
        plt.text(-0.5, offset, times[i], 
                fontsize=9, 
                verticalalignment='center',
                horizontalalignment='right')
    
    # Formatting
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Amplitude (μV) - Offset for visualization', fontsize=12)
    plt.title(f'EEG Signals in Cascade Form - First {num_signals} Measurements', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


filepath = r'C:\Users\msedo\Documents\CCEPs\Martin Garcia\Inomed M2\Data Export\CCEPs export W1 Fz.txt'

# Load all data
data, times, sr = load_inomed_eeg(filepath)
print(f"Shape:  {data.shape}, Sampling rate: {sr} Hz")
print(f"Time range: {times[0]} to {times[-1]}")

# Load with time filter
filtered_data, filtered_times, sr = load_inomed_eeg(
    filepath, 
    start_time="13:00:00", 
    end_time="14:00:00"
)
print(f"\nFiltered shape:  {filtered_data.shape}")
print(f"Filtered times: {filtered_times[0]} to {filtered_times[-1]}")

# Access channels
channel_1 = filtered_data[0]  # First channel
channel_2 = filtered_data[1]  # Second channel


plot_cascade(data, times, num_signals=5, spacing=2)