# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:53:42 2024

@author: Martina

FOR EVERY PATIENT AND CHANNEL, CHANGE FOLDER DIRECTORY ON LINE 31 AND EXCEL FILE EXPORT NAME ON LINE 324

"""

globals().clear()

from inomed.inoPatientData import *
from inomed.readEDF import *

import matplotlib.pyplot as plt
import mplcursors

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import find_peaks

import os
import glob

import seaborn as sns

# ---------------- Importing files from specified folder ----------------------

plt.close('all')

# Specify the folder containing the EDF files
folder = r'C:\Users\marti\OneDrive\Documents\UPC\Quart de carrera\8th Cuatrimestre\TFG\SJD\Data Recordings\PATIENT DATA\Patient 9\Channel 2-3'

# Get a list of all EDF files in the folder
files = glob.glob(os.path.join(folder, '*.edf'))

# Sort the files by modification time in ascending order
files_sorted = files.sort(key=os.path.getmtime)

# Create an array with file names
file_names = [os.path.basename(file) for file in files]

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Initialize an empty list to store signals
signals = []

# Initialize an empty list to store signals
f_signals = []

for i in np.array(range(0, len(file_names))):
    
    # load file
    ipd = readEDF(file=files[i])

    # Dictionary containing the metainformation stored in the EDF+ file
    info = ipd[0]
    
    # List of numpy.ndarray with measurement data
    data = ipd[1]
    
    # Number of records
    rec_num = ipd[0]['nRecords']
    
    # Number of signals
    sig_num = ipd[0]['nSignals']
    
    # Number of samples
    num_samples = ipd[0]['nSamples'][0]
    
    # Sampling frequency for each channel --> as  we use the same sampling frequency for all recordings, we select the first value in list
    fs = (num_samples/ipd[0]['durationRecords'])
    
    # Sampling period
    sample_range = range(num_samples)
    sample_array = np.array(sample_range)
    t = (sample_array / fs) * 1000
    
    y = -data[0]*1000
    
    signals.append(y)  # Store the signal in the list

    #---------------- Filtering data ------------------------------------------
    
    # Cutoff frequencies for the filters
    low_cutoff = 10.0  # Low-pass filter cutoff frequency in Hz
    high_cutoff = 1500.0  # High-pass filter cutoff frequency in Hz
    
    # Calculate the Nyquist frequency
    nyquist_freq = 0.5 * fs
    
    # Normalize cutoff frequencies to be between 0 and 1
    low_cutoff_norm = low_cutoff / nyquist_freq
    high_cutoff_norm = high_cutoff / nyquist_freq
    
    # Create a Butterworth filter
    order = 4  # Order of the filter
    b, a = signal.butter(order, [low_cutoff_norm, high_cutoff_norm], btype='band')
    
    # Apply the filter to the signal data
    filtered_y = signal.filtfilt(b, a, y)
    
    f_signals.append(y)
    
    
    # ---------------- Plotting results ---------------------------------------
    
    # Plot original signal on the first subplot
    ax1.plot(t, y, label='Original Signal')
    # Plot filtered signal on the second subplot
    ax2.plot(t, filtered_y, label='Filtered Signal')
    


ax1.set_ylabel("Amplitude [uV]")
ax1.set_title('Original Signal')
ax1.grid('minor')

ax2.set_ylabel("Amplitude [uV]")
ax2.set_xlabel("Time [ms]")
ax2.set_title('Filtered Signal')
ax2.grid('minor')

# Adjust layout to prevent overlap
plt.tight_layout()

# Add interactive cursors
mplcursors.cursor(ax1.lines + ax2.lines, hover=True)

plt.show()


# ------------------ Feature extraction: N1, P1, N2 ---------------------------

# Initialize lists to hold the data
data = {
    'Signal': [],
    'P1_Latency': [], 'P1_Amplitude': [],
    'N1_Latency': [], 'N1 matsumoto': [],
    'P2_Latency': [], 'P2_Amplitude': [],
    'N2_Latency': [], 'N2_Amplitude': []
}

# Plot the original signal
plt.figure(figsize=(10, 6))

# Plot each stored signal in the new figure
for idx in range(len(f_signals)):
    

    # Create masks for the sections
    mask_section1 = (t >= 10) & (t <= 27)
    mask_section2 = (t >= 40) & (t <= 90)
    mask_section3 = (t >= 20) & (t <= 50)
    mask_section4 = (t >= 6) & (t <= 9)  # New mask section

    # Extract the relevant sections
    section1 = f_signals[idx][mask_section1]
    section2 = f_signals[idx][mask_section2]
    section3 = f_signals[idx][mask_section3]
    section4 = f_signals[idx][mask_section4]  # Extract new section

    # Find minimum values in each section
    min_value_section1 = np.min(section1)
    min_value_section2 = np.min(section2)

    # Find the global index of the minimum value in section1
    global_indices_section1 = np.where(mask_section1)[0]
    min_idx_local_section1 = np.argmin(section1)
    min_idx_section1 = global_indices_section1[min_idx_local_section1]

    # Find the global index of the minimum value in section2
    global_indices_section2 = np.where(mask_section2)[0]
    min_idx_local_section2 = np.argmin(section2)
    min_idx_section2 = global_indices_section2[min_idx_local_section2]

    # Find maximum value in section 3
    max_value_section3 = np.max(section3)
    global_indices_section3 = np.where(mask_section3)[0]
    max_idx_local_section3 = np.argmax(section3)
    max_idx_section3 = global_indices_section3[max_idx_local_section3]

    # Find maximum value in the new section (P1)
    max_value_section4 = np.max(section4)
    global_indices_section4 = np.where(mask_section4)[0]
    max_idx_local_section4 = np.argmax(section4)
    max_idx_section4 = global_indices_section4[max_idx_local_section4]
    
    
    # Compute the coordinates of the intersection of the N1 vertical line with the P1-P2 line
    x1, y1 = t[max_idx_section4], max_value_section4
    x2, y2 = t[max_idx_section3], max_value_section3
    x3, y3 = t[min_idx_section1], min_value_section1

    # Equation of the line passing through (x1, y1) and (x2, y2)
    m = (y2 - y1) / (x2 - x1)  # Slope of the line
    c = y1 - m * x1            # Intercept of the line

    # Intersection point (x3, y4) where y4 is on the line
    y4 = m * x3 + c

    # Calculate Matsumoto N1 amplitude
    n1_matsumoto = y4 - y3

    # N2 amplitude
    n2_amp = max_value_section3 - min_value_section2

    
    # Append the values to the data lists
    data['Signal'].append(idx + 1)
    data['P1_Latency'].append(t[max_idx_section4])
    data['P1_Amplitude'].append(max_value_section4)
    data['N1_Latency'].append(t[min_idx_section1])
    data['N1 matsumoto'].append(n1_matsumoto)
    data['P2_Latency'].append(t[max_idx_section3])
    data['P2_Amplitude'].append(max_value_section3)
    data['N2_Latency'].append(t[min_idx_section2])
    data['N2_Amplitude'].append(n2_amp)
    
    
    plt.plot(t, f_signals[idx])
    
    # Plot the minimum values on the original signal with labels
    plt.scatter([t[mask_section1][np.argmin(f_signals[idx][mask_section1])], t[mask_section2][np.argmin(f_signals[idx][mask_section2])]],
                [min_value_section1, min_value_section2], color='red', zorder=2)

    # Plot the maximum value on the original signal with label
    plt.scatter(t[mask_section3][np.argmax(f_signals[idx][mask_section3])], max_value_section3, color='green', zorder=2)

    # Plot the maximum value in the new section (P1)
    plt.scatter(t[max_idx_section4], max_value_section4, color='blue', zorder=2)

    
    
    
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [uV]')
plt.title('Original Signal with Maximum and Minimum Values in Sections')
plt.legend() 
   
plt.grid('minor')
plt.show()

# Add mean values to the data
mean_data = {
    'Signal': ['Mean'],
    'P1_Latency': [np.mean(data['P1_Latency'])],
    'P1_Amplitude': [np.mean(data['P1_Amplitude'])],
    'N1_Latency': [np.mean(data['N1_Latency'])],
    'N1 matsumoto': [np.mean(data['N1 matsumoto'])],
    'P2_Latency': [np.mean(data['P2_Latency'])],
    'P2_Amplitude': [np.mean(data['P2_Amplitude'])],
    'N2_Latency': [np.mean(data['N2_Latency'])],
    'N2_Amplitude': [np.mean(data['N2_Amplitude'])]
}

# Add std values to the data
std_data = {
    'Signal': ['STD'],
    'P1_Latency': [np.std(data['P1_Latency'])],
    'P1_Amplitude': [np.std(data['P1_Amplitude'])],
    'N1_Latency': [np.std(data['N1_Latency'])],
    'N1 matsumoto': [np.std(data['N1 matsumoto'])],
    'P2_Latency': [np.std(data['P2_Latency'])],
    'P2_Amplitude': [np.std(data['P2_Amplitude'])],
    'N2_Latency': [np.std(data['N2_Latency'])],
    'N2_Amplitude': [np.std(data['N2_Amplitude'])]
}


# ------------ Averaged signal with standard deviation envelope ---------------

# Calculate the mean signal and standard deviation
mean_signal = np.mean(f_signals, axis=0)
std_deviation = np.std(f_signals, axis=0)

# Create masks for the sections
mask_section1 = (t >= 10) & (t <= 27)
mask_section2 = (t >= 20) & (t <= 50)
mask_section3 = (t >= 40) & (t <= 90)
mask_section4 = (t >= 6) & (t <= 12)  # New mask section

# Find minimum value in section 1
min_value_section1 = np.min(mean_signal[mask_section1])
global_indices_section1 = np.where(mask_section1)[0]
min_idx_local_section1 = np.argmin(mean_signal[mask_section1])
min_idx_section1 = global_indices_section1[min_idx_local_section1]

# Find maximum value in section 2
max_value_section2 = np.max(mean_signal[mask_section2])
global_indices_section2 = np.where(mask_section2)[0]
max_idx_local_section2 = np.argmax(mean_signal[mask_section2])
max_idx_section2 = global_indices_section2[max_idx_local_section2]

# Find minimum value in section 3
min_value_section3 = np.min(mean_signal[mask_section3])
global_indices_section3 = np.where(mask_section3)[0]
min_idx_local_section3 = np.argmin(mean_signal[mask_section3])
min_idx_section3 = global_indices_section3[min_idx_local_section3]

# Find maximum value in the new section (P1)
max_value_section4 = np.max(mean_signal[mask_section4])
global_indices_section4 = np.where(mask_section4)[0]
max_idx_local_section4 = np.argmax(mean_signal[mask_section4])
max_idx_section4 = global_indices_section4[max_idx_local_section4]

# Compute the coordinates of the intersection of the N1 vertical line with the P1-P2 line
x1, y1 = t[max_idx_section4], max_value_section4
x2, y2 = t[max_idx_section2], max_value_section2
x3, y3 = t[min_idx_section1], min_value_section1

# Equation of the line passing through (x1, y1) and (x2, y2)
m = (y2 - y1) / (x2 - x1)  # Slope of the line
c = y1 - m * x1            # Intercept of the line

# Intersection point (x3, y4) where y4 is on the line
y4 = m * x3 + c

# Calculate Matsumoto N1 amplitude
n1_matsumoto_avg = y4 - y3

# N2 amplitude
n2_amp_avg = max_value_section2 - min_value_section3

# Append N1, P1, and N2 values to the data table
data['Signal'].append('Averaged')
data['N1_Latency'].append(t[min_idx_section1])
data['N1 matsumoto'].append(n1_matsumoto_avg)
data['P1_Latency'].append(t[max_idx_section4])  # New P1
data['P1_Amplitude'].append(max_value_section4)  # New P1
data['P2_Latency'].append(t[max_idx_section2])
data['P2_Amplitude'].append(max_value_section2)
data['N2_Latency'].append(t[min_idx_section3])
data['N2_Amplitude'].append(n2_amp_avg)

# Plot the mean signal with standard deviation
plt.figure(figsize=(10, 6))
plt.plot(t, mean_signal, label='Averaged Signal')
plt.fill_between(t, mean_signal - std_deviation, mean_signal + std_deviation, color='gray', alpha=0.2, label='Standard Deviation')

# Plot the maximum and minimum values on the mean signal
plt.scatter(t[max_idx_section4], max_value_section4, color='orange', label='P1')
plt.scatter(t[min_idx_section1], min_value_section1, color='red', label='N1')
plt.scatter(t[max_idx_section2], max_value_section2, color='green', label='P2')
plt.scatter(t[min_idx_section3], min_value_section3, color='blue', label='N2')

# Add text labels beneath the legend
plt.text(80, 240, f'P1: lat.={t[max_idx_section4]:.2f} ms, amp.={max_value_section4:.2f} uV', ha='left')
plt.text(80, 220, f'N1: lat.={t[min_idx_section1]:.2f} ms, amp.={n1_matsumoto_avg:.2f} uV', ha='left')
plt.text(80, 200, f'P2: lat.={t[max_idx_section2]:.2f} ms, amp.={max_value_section2:.2f} uV', ha='left')
plt.text(80, 180, f'N2: lat.={t[min_idx_section3]:.2f} ms, amp.={n2_amp_avg:.2f} uV', ha='left')

plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (µV)')
plt.title('Averaged signal ± Standard deviation envelope')
plt.legend()
plt.xlim([0, 250])
plt.ylim([-200, 300])
plt.grid(True)
plt.show()



#----------------- Dataframe with extracted parameters ------------------------

# Append mean data to the dataframe
df = pd.DataFrame(data)

# Create a DataFrame for the mean data
mean_df = pd.DataFrame(mean_data)

# Create a DataFrame for the mean data
std_df = pd.DataFrame(std_data)

# Concatenate mean_df with df
df = pd.concat([df, mean_df], ignore_index=True)

# Concatenate std_df with df
df = pd.concat([df, std_df], ignore_index=True)

# Set pandas display options to show the full DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Display the DataFrame
print(df)

# Export DataFrame to Excel file
#df.to_excel(r'C:\Users\marti\OneDrive\Documents\UPC\Quart de carrera\8th Cuatrimestre\TFG\SJD\Data Recordings\PATIENT DATA\Patient 9\Channel 2-3 end\P11CH2-3_nofilt.xlsx', index=True)


#------------------- Statistical analysis -------------------------------------

# Definition of function to calculate frequencies for given variable in dataframe df


def calculate_frequency(data, column_name):
    # Absolute frequency
    ni = data[column_name].value_counts().sort_index()
    ni = ni.rename('ni')

    # Relative frequency
    fi = data[column_name].value_counts(normalize=True).sort_index()
    fi = fi.rename('fi')

    # Cumulative absolute frequency
    Ni = ni.cumsum()
    Ni = Ni.rename('Ni')

    # Cumulative relative frequency
    Fi = fi.cumsum()
    Fi = Fi.rename('Fi')

    # Concatenate the results into a DataFrame
    freq_data = pd.concat([ni, fi, Ni, Fi], axis=1)

    return freq_data


#freqs = calculate_frequency(df, 'N1_Latency')
#print(freqs)


# ------------------------ Histogram ---------------------------------------
'''
# Define the columns to plot
columns_to_plot = ['N1_Latency', 'P2_Latency', 'N2_Latency', 'N2_Amplitude', 'N1 matsumoto']

# Slice the DataFrame to exclude the last two rows
df_subset = df.iloc[:-2]

# Determine the number of subplots needed
num_columns = len(columns_to_plot)
num_rows = (num_columns + 1) // 2  # 2 columns per row

# Create subplots
fig, axes = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))
axes = axes.flatten()  # Flatten to easily iterate over

# Plot histograms and KDE lines for each specified column
for i, column in enumerate(columns_to_plot):
    sns.histplot(df_subset[column], bins=20, kde=True, color='skyblue', ax=axes[i])
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Distribution of {column}', fontsize=10)
    axes[i].grid(True)

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=3.0)
plt.show()


# ----------------------- Box plot ------------------------------------------

# Plotting boxplot for latency and amplitude variables in subplots
fig, axes = plt.subplots(2, 1, figsize=(5, 10))

# Boxplot for latency variables
sns.boxplot(data=df[['N1_Latency', 'P2_Latency', 'N2_Latency']], ax=axes[0])
axes[0].set_xlabel('Latency Variables')
axes[0].set_ylabel('Latency (ms)')
axes[0].set_title('Boxplot of Latency Variables', fontsize=14)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True)

# Boxplot for amplitude variables
sns.boxplot(data=df[['N1 matsumoto', 'P2_Amplitude', 'N2_Amplitude']], ax=axes[1])
axes[1].set_xlabel('Amplitude Variables')
axes[1].set_ylabel('Amplitude (uV)')
axes[1].set_title('Boxplot of Amplitude Variables', fontsize=14)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True)

plt.tight_layout(pad = 3.0)
plt.show()
'''



