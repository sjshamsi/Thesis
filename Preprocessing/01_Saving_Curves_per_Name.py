from pandas import read_parquet, concat
# from os.path import isfile

data_dir = '/home/shoaib/PSChallenge/'

# %%
all_lightcurves = read_parquet(data_dir + 'all_lightcurves.parquet')

# %%
names_g = set(all_lightcurves[all_lightcurves['band'] == 'g']['name'].to_numpy())
names_r = set(all_lightcurves[all_lightcurves['band'] == 'r']['name'].to_numpy())

both_names = names_g.intersection(names_r)
only_g_names = names_g - names_r
only_r_names = names_r - names_g

# %%
for name in both_names:
    name_lcs = all_lightcurves[all_lightcurves['name'] == name]
    name_lcs.drop(columns=['name'])

    save_path = data_dir + f"Lightcurves_by_Name/{name}.csv"
    
    # if isfile(save_path):
    #     print(f'{save_path} exists already. Skipping...')
    # else:
    #     name_lcs.to_csv(save_path, index=False)
    #     print(f'{name}.csv saved to {save_path}')
    name_lcs.to_csv(save_path, index=False)
    print(f'{name}.csv saved to {save_path}')

for name in only_g_names:
    name_lcs = all_lightcurves[all_lightcurves['name'] == name]
    name_lcs.drop(columns=['name'])

    save_path = data_dir + f"Lightcurves_by_Name/only_g/{name}.csv"
    
    # if isfile(save_path):
    #     print(f'{save_path} exists already. Skipping...')
    # else:
    #     name_lcs.to_csv(save_path, index=False)
    #     print(f'{name}.csv saved to {save_path}')
    name_lcs.to_csv(save_path, index=False)
    print(f'{name}.csv saved to {save_path}')

for name in only_r_names:
    name_lcs = all_lightcurves[all_lightcurves['name'] == name]
    name_lcs.drop(columns=['name'])

    save_path = data_dir + f"Lightcurves_by_Name/only_r/{name}.csv"
    
    # if isfile(save_path):
    #     print(f'{save_path} exists already. Skipping...')
    # else:
    #     name_lcs.to_csv(save_path, index=False)
    #     print(f'{name}.csv saved to {save_path}')
    name_lcs.to_csv(save_path, index=False)
    print(f'{name}.csv saved to {save_path}')
