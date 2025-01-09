#import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc
from scipy.signal import find_peaks
from scipy.signal import correlate

data_dir = '/home/shoaib/PSChallenge/'

#load up filenames
files=[]
for filename in os.listdir(data_dir + 'Lightcurves_by_Name/'):
  if filename.endswith('.csv'):
    files.append(filename)
#print(files)

features = ['name', 'rms_g', 'rms_r', 'amplitude_g', 'amplitude_r', 'mag_std_g', 'mag_std_r', 'f_var_g', 'f_var_r', 'color_index', 'peak_lag', 'peak_correlation']
df_features = pd.DataFrame(columns=features)

#METHODS TO COMPUTE FEATURES

# RMS variability

def calculate_rms_variability(light_curve):
    """
    Calculate RMS variability for a light curve.
    
    Parameters:
    light_curve (pd.DataFrame): A DataFrame containing 'mjd', 'mag', and 'magerr' columns.
    
    Returns:
    float: The RMS variability of the light curve.
    """
    # Extract magnitudes and magnitude errors
    mag = light_curve['mag']
    magerr = light_curve['magerr']
    
    # Weights are inversely proportional to the square of the magnitude errors
    weights = 1 / magerr**2
    
    # Calculate the weighted mean of the magnitudes
    weighted_mean_mag = np.sum(weights * mag) / np.sum(weights)
    
    # Calculate deviations from the weighted mean
    deviations = mag - weighted_mean_mag
    
    # Compute the RMS variability
    rms_variability = np.sqrt(np.sum(weights * deviations**2) / np.sum(weights))
    
    return rms_variability

# PEAK-TO-PEAK AMPLITUDE

def calculate_peak_to_peak_amplitude(light_curve):
    """
    Calculate the peak-to-peak amplitude of a light curve.
    
    Parameters:
    light_curve (pd.DataFrame): A DataFrame containing 'mjd', 'mag', and 'magerr' columns.
    
    Returns:
    float: The peak-to-peak amplitude of the light curve.
    """
    # Extract magnitudes
    mag = light_curve['mag'].values
    
    # Find peaks (local maxima)
    peaks, _ = find_peaks(mag)
    
    # Find valleys (local minima) by inverting the magnitude
    valleys, _ = find_peaks(-mag)
    
    if len(peaks) == 0 or len(valleys) == 0:
        # If no peaks or valleys are detected, return 0 as amplitude
        return 0
    
    # Get the highest peak and the lowest valley
    max_peak = mag[peaks].max()
    min_valley = mag[valleys].min()
    
    # Calculate the peak-to-peak amplitude
    amplitude = max_peak - min_valley
    
    return amplitude

# F-VARIABILITY

#method

def calculate_f_variability(light_curve):
    """
    Calculate the F-variability of a light curve.
    
    Parameters:
    light_curve (pd.DataFrame): A DataFrame containing 'mjd', 'mag', and 'magerr' columns.
    
    Returns:
    float: The F-variability of the light curve. If the result is invalid, returns NaN.
    """
    
    # Extract magnitudes and magnitude errors
    mag = light_curve['mag'].values
    magerr = light_curve['magerr'].values
    
    # Mean magnitude
    mean_mag = np.mean(mag)
    
    if mean_mag == 0:
        # Avoid division by zero
        return np.nan
    
    # Variance of the magnitudes
    variance_mag = np.var(mag, ddof=1)  # Sample variance (unbiased)
    
    # Mean squared error
    mean_squared_error = np.mean(magerr**2)
    
    # Corrected variance (subtract mean squared error from variance)
    corrected_variance = variance_mag - mean_squared_error
    
    if corrected_variance < 0:
        # If corrected variance is negative, F-var is not defined
        return np.nan
    
    # F-variability
    f_variability = np.sqrt(corrected_variance) / mean_mag
    
    return f_variability


# COLOUR INDEX

#method

def calculate_color_index(lc_g, lc_r):
    """
    Calculate the color index from g-band and r-band light curves.
    
    Parameters:
    lc_g (pd.DataFrame): A DataFrame containing 'mjd', 'mag', and 'magerr' for the g-band.
    lc_r (pd.DataFrame): A DataFrame containing 'mjd', 'mag', and 'magerr' for the r-band.
    
    Returns:
    float: The color index (mean_g - mean_r).
    """
    # Calculate the mean magnitude in each band
    mean_g = np.mean(lc_g['mag'])
    mean_r = np.mean(lc_r['mag'])
    
    # Compute the color index
    color_index = mean_g - mean_r
    
    return color_index

# PEAK LAG AND CORRELATION

def compute_peak_lag_and_correlation(lc_g, lc_r, max_lag=100, num_points=500):
    """
    Compute the peak lag and peak correlation for g-band and r-band light curves.
    
    Parameters:
        lc_g (DataFrame): g-band light curve with 'mjd' and 'mag' columns.
        lc_r (DataFrame): r-band light curve with 'mjd' and 'mag' columns.
        max_lag (int): Maximum lag in days for cross-correlation.
        num_points (int): Number of points for interpolation.
    
    Returns:
        float: Peak lag time (days).
        float: Peak correlation value.
    """
    # Interpolate magnitudes onto a common grid of times
    common_times = np.linspace(
        max(lc_g['mjd'].min(), lc_r['mjd'].min()),
        min(lc_g['mjd'].max(), lc_r['mjd'].max()),
        num=num_points
    )
    mag_g_interp = np.interp(common_times, lc_g['mjd'], lc_g['mag'])
    mag_r_interp = np.interp(common_times, lc_r['mjd'], lc_r['mag'])

    # Subtract the mean to normalize for cross-correlation
    mag_g_interp -= np.mean(mag_g_interp)
    mag_r_interp -= np.mean(mag_r_interp)

    # Compute cross-correlation
    correlation = correlate(mag_g_interp, mag_r_interp, mode='full')
    lags = np.arange(-len(common_times) + 1, len(common_times))

    # Limit lags to the max_lag range
    valid_lags = (lags >= -max_lag) & (lags <= max_lag)
    lags = lags[valid_lags]
    correlation = correlation[valid_lags]

    # Find the lag corresponding to the peak correlation
    peak_idx = np.argmax(correlation)
    peak_lag = lags[peak_idx]
    peak_correlation = correlation[peak_idx]

    return peak_lag, peak_correlation

file_num = 0

for file_num in range(0,len(files)):
    print('Computing features for file number: ', file_num, ' of: ', len(files))
    filename=files[file_num]
    this_lc=pd.read_csv('Lightcurves_by_Name/'+filename, index_col=False)
    #retreive name of the LC
    lc_name = filename.rsplit('.csv', 1)[0]
    print('Object name: ', lc_name)

    #COMPUTE THE FEATURES FOR THIS_LC
    # Extract unique 'oid_alerce' values for each band
    oid_g = this_lc.loc[this_lc['band'] == 'g', 'oid_alerce'].unique().tolist()
    oid_r = this_lc.loc[this_lc['band'] == 'r', 'oid_alerce'].unique().tolist()
    
    #create dataframes for g and r (for multiple light curve observations - we ignore the gaps)
    # WE ALSO NEED TO SORT BY MJD
    this_lc_g = this_lc[this_lc['oid_alerce'].isin(oid_g)].reset_index(drop=True)
    this_lc_r = this_lc[this_lc['oid_alerce'].isin(oid_r)].reset_index(drop=True)

    #keep only necessary columns to extract features
    lc_g = this_lc_g[['mjd','mag', 'magerr']].sort_values(by=['mjd'], ascending=True)
    lc_r = this_lc_r[['mjd','mag', 'magerr']].sort_values(by=['mjd'], ascending=True)

    # calculate RMS variability in both bands
    rms_g = calculate_rms_variability(lc_g[['mjd', 'mag', 'magerr']])
    rms_r = calculate_rms_variability(lc_r[['mjd', 'mag', 'magerr']])

    #calculate peak-to-peak amplitude
    amplitude_g = calculate_peak_to_peak_amplitude(lc_g[['mjd', 'mag', 'magerr']])
    amplitude_r = calculate_peak_to_peak_amplitude(lc_r[['mjd', 'mag', 'magerr']])

    # standard deviation
    mag_std_g = lc_g['mag'].std()
    mag_std_r = lc_r['mag'].std()

    #F-variability
    f_var_g = calculate_f_variability(lc_g[['mjd', 'mag', 'magerr']])
    f_var_r = calculate_f_variability(lc_r[['mjd', 'mag', 'magerr']])

    #colour index
    color_index = calculate_color_index(lc_g[['mjd', 'mag', 'magerr']], lc_r[['mjd', 'mag', 'magerr']])

    #peak lag and correlation
    peak_lag, peak_correlation = compute_peak_lag_and_correlation(
        lc_g[['mjd', 'mag', 'magerr']],
        lc_r[['mjd', 'mag', 'magerr']],
        max_lag=10
    )

    # Add all features into a dataframe
    this_features = [lc_name, rms_g, rms_r, amplitude_g, amplitude_r, mag_std_g, mag_std_r, f_var_g, f_var_r, color_index, peak_lag, peak_correlation]
    df_features.loc[len(df_features)] = this_features

df_features.to_csv(data_dir + 'new_features.csv', index=False)

print('All object features have been computed. Saved in new_features.csv')
