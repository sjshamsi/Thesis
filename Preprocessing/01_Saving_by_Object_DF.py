# %%
### Import stuff
from pandas import read_csv, Series
from numpy import sum, isnan

# %%
data_dir = '/home/shoaib/PSChallenge/'

# %%
features_by_lc = read_csv(data_dir + 'gr_features_by_oid.csv')

features_by_lc_g = features_by_lc[features_by_lc['band'] == 'g']
features_by_lc_r = features_by_lc[features_by_lc['band'] == 'r']

# %%
### Let's create a DF to store our OIDs and types

name_type_df = features_by_lc[['name', 'type']]
name_type_df = name_type_df.drop_duplicates(subset=['name'], keep='first', ignore_index=True)

features_by_lc_g = features_by_lc_g.drop(columns=['type', 'band', 'oid_alerce'])
features_by_lc_r = features_by_lc_r.drop(columns=['type', 'band', 'oid_alerce'])

# %%

def weighted_mean(series, weights):
    series = series.to_numpy()
    weights = weights.to_numpy()
    normalised_weights = weights / sum(weights)
    
    mask = ~isnan(series)
    return (series[mask] * normalised_weights[mask]).sum() / normalised_weights[mask].sum()

features_by_object_g = features_by_lc_g.groupby('name', as_index=False).apply(lambda group: Series({col: weighted_mean(group[col], group['n_good_det']) for col in features_by_lc_g.columns[1:]}))
features_by_object_r = features_by_lc_r.groupby('name', as_index=False).apply(lambda group: Series({col: weighted_mean(group[col], group['n_good_det']) for col in features_by_lc_g.columns[1:]}))

# %%
features_by_object_g.columns = ['name'] + [f"{col}_g" for col in features_by_object_g.columns[1:]]
features_by_object_r.columns = ['name'] + [f"{col}_r" for col in features_by_object_r.columns[1:]]

features_by_object_g = features_by_object_g.rename(columns={'n_good_det_g': 'avg_good_det_g'})
features_by_object_r = features_by_object_r.rename(columns={'n_good_det_r': 'avg_good_det_r'})

# %%
features_by_object = features_by_object_g.merge(features_by_object_r, on='name', how='inner')

features_by_object = features_by_object.merge(name_type_df, on='name', how='left')
features_by_object_g = features_by_object_g.merge(name_type_df, on='name', how='left')
features_by_object_r = features_by_object_r.merge(name_type_df, on='name', how='left')

# %% [markdown]
# Let's save all our files now.

# %%
features_by_object.to_csv(data_dir + 'gr_features_by_object.csv', index=False)
features_by_object_g.to_csv(data_dir + 'g_features_by_object.csv', index=False)
features_by_object_r.to_csv(data_dir + 'r_features_by_object.csv', index=False)
