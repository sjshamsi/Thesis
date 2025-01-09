# %%
from os import chdir
chdir('/home/shoaib/Thesis/dmdt_Analysis/')
from dmdt_functions import get_differenciation, get_2Dhistogram

# %%
from pandas import read_parquet, DataFrame, concat
from numpy import linspace

# %%
filtered_lightcurves = "/home/shoaib/PSChallenge/filtered_lightcurves.parquet"
original_features_by_oid = "/home/shoaib/PSChallenge/original_features_by_oid.csv"

xbins, ybins = 51, 51
# These files do'nt exist yet, they are save paths for later in the notebook
dmdt_by_name_band_50x50 = '/home/shoaib/Thesis/dmdt_Analysis/Histograms/dmdt_by_Name_Band_50x50.pkl'

# %%
### By experimentation we know that these are good bins
log_dt_bins = linspace(-4, 3.5, xbins)
dm_bins = linspace(-0.85, 0.8, ybins)

all_lightcurves = read_parquet(filtered_lightcurves)
# all_lightcurves = all_lightcurves.sample(frac=0.1)

# %%
### Looping now
grouped = all_lightcurves.groupby(['name', 'band'])
histogram_dict_list = []
num_total_groups = len(grouped)
count = 0

for (name, band), df in grouped:
    count += 1
    print(f'{count}\t/ {num_total_groups} objects in their bands done.')
    
    df = df.sort_values(by='mjd')
    times, mags = df['mjd'].to_numpy(), df['mag'].to_numpy()
    dtimes, dmags = get_differenciation(times=times, magnitudes=mags)
    hist, _junk, _junk = get_2Dhistogram(dtimes=dtimes, dmagnitudes=dmags, dt_bins=log_dt_bins, dm_bins=dm_bins, normalise=False, scale_factor=False)
    
    if hist.sum() == 0:
        continue
    
    hist_normalised = hist / hist.sum()
    object_type = df['type'].iloc[0]
    n_good_det = len(df)
    
    histogram_dict = {'name': name, 'type': object_type, 'band': band, 'n_good_det': n_good_det,
                      'histogram': hist, 'histogram_normalised': hist_normalised}
    histogram_dict_list.append(histogram_dict)
    

# %%
hist_df = DataFrame(histogram_dict_list)

# %%
grouped = hist_df.groupby('name')

for name, df in grouped:
    if len(df) < 2:
        continue
    
    hist = df['histogram'].sum()
    hist_normalised = hist / hist.sum()
    n_good_det = df['n_good_det'].sum()
    object_type = df['type'].iloc[0]

    histogram_dict = {'name': name, 'type': object_type, 'band': 'combined', 'n_good_det': n_good_det,
                      'histogram': hist, 'histogram_normalised': hist_normalised}
    histogram_dict_list.append(histogram_dict)

# %%
hist_df = DataFrame(histogram_dict_list)
hist_df = hist_df.sort_values(by=['type', 'name', 'band'], ignore_index=True)

# %%
hist_df.reset_index(drop=True).to_pickle(dmdt_by_name_band_50x50)

# %%