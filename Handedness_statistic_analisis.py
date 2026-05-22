# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:59:04 2024

@author: Martina
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

from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind

plt.close('all')

def merge_excel_files_in_folder(folder_path):
    # Get all Excel files in the specified folder
    excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through each Excel file
    for file in excel_files:
        # Read the Excel file into a DataFrame, ignoring the last three rows and first two columns
        df = pd.read_excel(file, skipfooter=3).iloc[:, 2:]

        # Append the DataFrame to the list
        dfs.append(df)

    # Merge all DataFrames in the list
    merged_df = pd.concat(dfs, ignore_index=True)

    return merged_df

folder_path_R = r'C:\Users\msedo\Documents\CCEPs\Analysis data\Handedness\R'
R = merge_excel_files_in_folder(folder_path_R)
R = R[(R['N1 matsumoto'] >= 0) & (R['N1 matsumoto'] <= 400) & (R['N2_Amplitude'] <= 400) & (R['N1_Latency'] <= 48)]

folder_path_L = r'C:\Users\msedo\Documents\CCEPs\Analysis data\Handedness\L'
L = merge_excel_files_in_folder(folder_path_L)
L = L[(L['N1 matsumoto'] >= 0) & (L['N1 matsumoto'] <= 400) & (L['N2_Amplitude'] <= 400) & (L['N1_Latency'] <= 48)]

# ------------------------------- Mean, median, std -------------------------------------------


def compute_statistics(df):
    # Define the variables of interest
    variables = ['N1_Latency', 'N1 matsumoto', 'P2_Latency', 'N2_Latency', 'N2_Amplitude']
    
    # Compute the mean, median, and standard deviation for each variable
    stats = {var: [np.mean(df[var]), np.median(df[var]), np.std(df[var])] for var in variables}

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(stats, index=['Mean', 'Median', 'Standard Deviation'])
    
    # Transpose the DataFrame for better readability
    results_df = results_df.T
    
    # Display the results
    print(results_df)

print("Statistics of R dataset: ")
compute_statistics(R)

print("Statistics of L dataset: ")
compute_statistics(L)

############################### Graphic analysis ##########################################

# ----------------------------- Histogram function --------------------------------------

def plot_histograms(df, df_name):
    """
    Generates histograms with KDE lines for specified columns in a given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    df_name (str): The name of the DataFrame.
    """
    # Define the columns to plot
    columns_to_plot = ['N1_Latency', 'P2_Latency', 'N2_Latency', 'N2_Amplitude', 'N1 matsumoto']

    # Determine the number of subplots needed
    num_columns = len(columns_to_plot)
    num_rows = (num_columns + 1) // 2  # 2 columns per row

    # Create subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))
    axes = axes.flatten()  # Flatten to easily iterate over

    # Plot histograms and KDE lines for each specified column
    for i, column in enumerate(columns_to_plot):
        sns.histplot(df[column], bins=20, kde=True, color='skyblue', ax=axes[i])
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Distribution of {column} in {df_name}', fontsize=10)
        axes[i].grid(True)

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.0)
    plt.show()


plot_histograms(R, 'R')

plot_histograms(L, 'L')

# --------------------------------- Violin Plot function ------------------------------

# Function to calculate the mean from a bootstrap sample
def bootstrap_mean(data):
    bootstrap_means = []
    n_bootstraps = 1000

    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])

    return confidence_interval

def plot_combined_violin(variable_name, svar, R, L, units='units'):
    """
    Generate combined violin plots for a variable across two DataFrames.
    
    Parameters:
    variable_name (str): Name of the variable (column) to plot.
    group_name (str): Name of the grouping variable (e.g., 'Age Group').
    ag2 (pd.DataFrame): DataFrame containing data for the first group (e.g., '5 - 8 years').
    ag3 (pd.DataFrame): DataFrame containing data for the second group (e.g., '>8 years').
    
    Returns:
    None
    """
    # Prepare data for plotting from ag2
    data_R = {
        svar: ['R'] * len(R[variable_name]),
        variable_name: R[variable_name]
    }
    df_R = pd.DataFrame(data_R)
    
    # Prepare data for plotting from ag3
    data_L = {
        svar: ['L'] * len(L[variable_name]),
        variable_name: L[variable_name]
    }
    df_L = pd.DataFrame(data_L)
    
    # Combine data into a single DataFrame for plotting
    df_combined = pd.concat([df_R, df_L], ignore_index=True)
    
    # Create a single figure for violin plots
    plt.figure(figsize=(12, 8))
    
    # Set the same font size for all text elements
    font_size = 20
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)  # legend fontsize
    
    # Plot violin plot for both datasets on the same figure
    ax = sns.violinplot(x=svar, y=variable_name, data=df_combined, hue=svar, linewidth=2.5)
    
    # ---------------------------------------------------------------------------
    # Calculate confidence intervals for each group and plot on the figure
    groups = [R, L]
    diagnoses = ['R', 'L']
    for i, group in enumerate(groups):
        sample_data = group[variable_name]
        confidence_interval = bootstrap_mean(sample_data)

        # Plot confidence interval on the figure
        x_loc = i
        y_loc = ax.get_ylim()[1]*0.95
        plt.text(x_loc, y_loc, f"95% CI: {confidence_interval[0]:.2f} - {confidence_interval[1]:.2f}", fontsize=20, ha='center')

    # ---------------------------------------------------------------------------    
    # Calculate p-values for each combination of groups
    p_values = []
    
    # Adjust y_loc base for multiple p-values
    y_loc_base = df_combined[variable_name].max()
    y_step = (df_combined[variable_name].max() - df_combined[variable_name].min()) * 0.05
    x_loc = 0.5
    #plt.text(x_loc, y_loc_base + y_step, "p-values:", fontsize=10, va='top')
    
    
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            stat, p_value = mannwhitneyu(groups[i][variable_name], groups[j][variable_name], alternative='two-sided')
            
            p_value = f"{p_value:.5f}"
            p_values.append({
                'Group 1': diagnoses[i],
                'Group 2': diagnoses[j],
                'p-value': p_value
            })
            
            # Plot p-value in the top right corner of the figure
            y_loc = y_loc_base - (len(p_values) - 1) * y_step
            '''
            p_value_str = f"{p_value:.3f}"
            if p_value < 0.05:
                p_value_str += "*"
            plt.text(x_loc, y_loc, f"{diagnoses[i]} vs {diagnoses[j]}: {p_value_str}", fontsize=10, va='top')
'''
    # Create a DataFrame for the p-values
    p_values_df = pd.DataFrame(p_values)
    
    order = df_combined[svar].unique()

    # ---------------------------------------------------------------------------
    # Plot the counts above the corresponding violin plot
    for c, d in enumerate(order):
        n = df_combined[svar].value_counts().get(d, 0)
        plt.text(c+0.1, ax.get_ylim()[1] * 0.9, f"n={n}")
    
    #plt.title(f'Violin Plot of {variable_name} for {svar}')
    plt.xlabel(f'{svar}')
    plt.ylabel(f'{variable_name} ({units})')  # Replace with appropriate units if known
    plt.tight_layout()
    plt.show()
    
    print()
    print(f'P-values for {variable_name}:')
    print(p_values_df)


# --------------------------- N1 Latency Violin plots ---------------------------------

plot_combined_violin('N1_Latency','Handedness', R, L, units='ms')


# --------------------------- N1 amplitude Violin plots ---------------------------------

plot_combined_violin('N1 matsumoto','Handedness', R, L, units='µV')

# --------------------------- P2 latency Violin plots ---------------------------------

plot_combined_violin('P2_Latency','Handedness', R, L, units='ms')

# --------------------------- N2 latency Violin plots ---------------------------------

plot_combined_violin('N2_Latency','Handedness', R, L, units='ms')

# --------------------------- N2 amplitude Violin plots ---------------------------------

plot_combined_violin('N2_Amplitude','Handedness', R, L, units='µV')

# ------------------------------- Significance tests (p-value) -------------------------------

