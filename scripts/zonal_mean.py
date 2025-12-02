# %%
# %load_ext autoreload
# %autoreload 2
from ectools import ecio
from matplotlib.colors import LogNorm, NoNorm
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import data_paths as dp


# %% Load multiple files
filelist_july = ecio.get_filelist(
    basedir=dp.ACMCAP,
    start_range_str="20251101T000000Z",
    end_range_str="20251130T235959Z",
    product_baseline="BC",
    production_centre="EX",
    product_type="ACM_CAP_2B",
    nested_directory_structure=True,
)
len(filelist_july)


# %%
def preprocess(
    ds,
    data_vars=[
        # "ice_normalized_number_concentration",
        # "ice_N0prime",
        # "ice_extinction",
        "ice_water_path",
        # "ice_water_path_error",
        # "retrieval_status",
        # "hydrometeor_classification",
    ],
    coords=["latitude", "longitude", "time", "height"],
    get_xmet=False,
):

    ds = ecio.trim_to_frame(ds, verbose=False)
    # ds = ds.isel(JSG_height=slice(241))
    if len(data_vars) != 0:
        ds = ds[data_vars + coords]

    if get_xmet:
        orbit_frame = ds.encoding["source"].split("/")[-1].split("_")[-1].split(".")[0]
        frame = orbit_frame[-1]
        orbit = orbit_frame[:-1]
        XMET = ecio.load_XMET(dp.XMET, orbit=orbit, frame=frame)
        XMET.close()
        ds = ecio.get_XMET(
            XMET, ds, XMET_1D_variables=[], XMET_2D_variables=["temperature"]
        )

    return ds.set_coords(coords)


ds_all = xr.open_mfdataset(
    filelist_july[:],
    group="ScienceData",
    combine="nested",
    concat_dim=["along_track"],
    parallel=True,
    chunks="auto",
    compat="broadcast_equals",
    preprocess=preprocess,
)

#%%
if set_clearsky_to_zero:=True:
    mask_clearsky = (ds_all["retrieval_status"]==-2).all(dim='CPR_height')
    ds_all = ds_all.where(~mask_clearsky, other=0.0)

# %% group by latitude bins and calculate zonal mean
lat_bins = np.arange(-90, 91, 5)
lat_bin_mids = (lat_bins[:-1] + lat_bins[1:]) / 2
groups_iwp = ds_all[["ice_water_path", "latitude"]].groupby(
    {
        "latitude": xr.groupers.BinGrouper(
            bins=lat_bins, labels=lat_bin_mids, include_lowest=True, right=True
        )
    }
)

zonal_mean_iwp = groups_iwp.mean(dim="along_track")["ice_water_path"].compute()

# %%
zonal_mean_iwp.plot(x="latitude_bins", marker="^", figsize=(12, 6))
plt.grid(True)
plt.title("Zonal Mean Ice Water Path - July 2025")

# %%
if save:=True:
    zonal_mean_iwp.to_netcdf(
        os.path.join(
            "/home/anqil/arts_ir_simulation/data/earthcare",
            "zonal_mean_iwp_acmcap_nov_2025.nc",
        )
    )

# %%
lat_bins = np.arange(-90, 91, 5)
lat_bin_mids = (lat_bins[:-1] + lat_bins[1:]) / 2

zonal_mean_dardar = xr.open_dataset(
    "/scratch/may/AtmIceStatus/dardar_FWP_PDF_zonal_mean_pre2010_july.nc"
)
zonal_mean_aws_ccic = (
    xr.open_dataset("/scratch/evoy/aws_ccic_fwp_binned_global_july_v2.nc")
    .assign_coords(lat_bin=lat_bin_mids)
    .sel(lat_bin=slice(-61, 61))
)
zonal_mean_ec_ccic = xr.open_dataset(
    "/scratch/may/data/CCIC_on_earthCARE_fwp_zonal_mean_july_2025.nc"
)
zonal_mean_acmcap = xr.open_dataset(
    "/home/anqil/arts_ir_simulation/data/earthcare/zonal_mean_iwp_july_2025.nc"
)

zonal_mean_ccld = xr.open_dataset(
    "/home/anqil/arts_ir_simulation/data/earthcare/zonal_mean_iwp_ccld_july_2025.nc"
)
# set color cycle and line styles
colors = ["#e69f00", "#56b4e9", "#009e73", "#cc79a7", "#637A87", "#D55E00"]
linestyles = ["-", "-", "-.", "--","--","-."]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors, linestyle=linestyles)

plt_kws = {"alpha":0.8, "linewidth": 5}
fig, ax = plt.subplots(1,2,figsize=(18, 6), sharey=True)
zonal_mean_acmcap["ice_water_path"].plot(ax=ax[0], label="ACM-CAP-2B:BA", x="latitude_bins", **plt_kws)
zonal_mean_ccld["ice_water_path"].plot(ax=ax[0], label="CPR-CLD-2A:BA", x="latitude_bins", **plt_kws)
zonal_mean_aws_ccic["fwp_mean"].plot(ax=ax[0], label="AWS", x="lat_bin", **plt_kws)
zonal_mean_aws_ccic["fwp_ccic"].plot(ax=ax[0], label="CCIC on AWS path", x="lat_bin", **plt_kws)
zonal_mean_ec_ccic["zonal_mean"].plot(ax=ax[0], label="CCIC on EarthCARE path", x="latitude", **plt_kws)
zonal_mean_dardar["FWP_zonal_mean"].plot(ax=ax[0], label="DARDAR (2007-2010)", x="latitude_bin_centres", **plt_kws)

ax[0].set_title("Zonal Mean IWP - July 2025")
ax[0].set_xlabel(r"Latitude (°)")
ax[0].set_ylabel(r"IWP $(kgm^{-2})$")
ax[0].grid(True)
ax[0].legend(fontsize=12)


zonal_mean_acmcap_nov = xr.open_dataset(
    "/home/anqil/arts_ir_simulation/data/earthcare/zonal_mean_iwp_acmcap_nov_2025.nc"
)
    
zonal_mean_acmcap_nov["ice_water_path"].plot(ax=ax[1], label="ACM-CAP-2B:BC", x="latitude_bins", **plt_kws)
ax[1].set_title("Zonal Mean IWP - Nov 2025")
ax[1].set_xlabel(r"Latitude (°)")
ax[1].set_ylabel(r"IWP $(kgm^{-2})$")
ax[1].grid(True)
ax[1].legend(fontsize=16)
# %%
