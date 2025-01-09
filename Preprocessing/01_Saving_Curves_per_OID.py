from pandas import read_parquet

data_dir = '/home/shoaib/PSChallenge/'

ztf_g = read_parquet(data_dir + 'filtered_QSO_AGN_Blazar_ZTF_DR6_lcs_gband.parquet')
ztf_r = read_parquet(data_dir + 'filtered_QSO_AGN_Blazar_ZTF_DR6_lcs_rband.parquet')

g_group = ztf_g.groupby('oid_alerce')
r_group = ztf_r.groupby('oid_alerce')

for group in [g_group, r_group]:
    for group_name, group_data in group:
        file_name = f"{data_dir+'Lightcurves_by_OID/'}{group_name}.csv"
        group_data.to_csv(file_name, index=False)
        print(f"Saved group '{group_name}' to {file_name}")
