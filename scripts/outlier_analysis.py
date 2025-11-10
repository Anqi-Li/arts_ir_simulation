# %%
# %load_ext autoreload
# %autoreload 2

from analysis import (
    load_and_merge_ec_data,
    load_arts_results,
    load_and_merge_acmcap_data,
    get_bias,
    get_var_at_max_height_by_dbz,
    get_difference_by_dbzs,
)
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from earthcare_ir import Habit, PSD
from matplotlib.colors import LogNorm
import data_paths as dp
import os
from earthcare_ir import PSD, Habit
from ectools import colormaps
from ectools import ecio

save = False

# %% select orbit_frames
# check the list of orbit frames in the arts output data
print("Getting list of orbit frames from ARTS output data...")
path_to_search_orbit_frames = dp.arts_output_TIR2
print("ARTS output data path:", path_to_search_orbit_frames)
file_paths = glob.glob(os.path.join(path_to_search_orbit_frames, f"*{Habit.Plate}_{PSD.F07T}.nc"))
orbit_frames_arts = [f.split("/")[-1].split("_")[2] for f in file_paths]
print(f"Number of orbit frames in ARTS output data: {len(orbit_frames_arts)}")

# check the list of orbit frames in the ACMCAP data
filelist_acmcap = ecio.get_filelist(
    basedir=dp.ACMCAP,
    product_baseline="BA",
    production_centre="EX",
    nested_directory_structure=True,
)
orbit_frames_acmcap = [f.split("/")[-1].split("_")[-1].split(".")[0] for f in filelist_acmcap]
overlap_orbit_frames = set(orbit_frames_arts) & set(orbit_frames_acmcap)
print(f"Number of overlapping orbit frames between ARTS and ACMCAP: {len(overlap_orbit_frames)}")

orbit_frames = sorted(list(overlap_orbit_frames))

if shorten_to_test := True:
    print("Shorten the list of orbit frames for testing...")
    orbit_frames = orbit_frames[:600]  # for testing purpose
    print(f"Number of orbit frames used for testing: {len(orbit_frames)}")
    save = False

# %% load and merge arts and EC data
from tqdm import tqdm
import io
from contextlib import redirect_stdout, redirect_stderr

# orbit_frame = overlap_orbit_frames
data = []
orbit_failed = []
for o in tqdm(orbit_frames, desc="Loading ARTS results", total=len(orbit_frames)):
    try:
        # Load ARTS results
        ds_arts = load_arts_results(habit=Habit.Column, psd="*", orbit_frame=o)
        # Merge with EC data
        ds_arts_and_ec = load_and_merge_ec_data(ds_arts).assign(orbit_frame=o)
        # Merge with ACMCAP data
        _buf = io.StringIO()  # avoid printing to stdout
        with redirect_stdout(_buf), redirect_stderr(_buf):
            ds_arts_and_ec_acm = load_and_merge_acmcap_data(
                ds_arts=ds_arts_and_ec,
                orbit_frame=o,
                keep_vars=[
                    "MSI_longwave_brightness_temperature",
                    "MSI_longwave_brightness_temperature_forward",
                    "liquid_optical_depth",
                ],
                trim=True,
            )
        data.append(ds_arts_and_ec_acm)
    except Exception as e:
        print(f"Error loading ARTS output for orbit frame {o}: {e}")
        orbit_failed.append((o, e))
        continue

if "along_track" in data[0].coords and "along_track" not in data[0].dims:
    data = [ds.swap_dims(along_track="along_track") for ds in data]
ds = xr.concat(data, dim="along_track")
number_of_frames = len(data)
print(f"Total number of loaded orbit frames: {number_of_frames}")

# mask profiles with all low reflectivity
mask = (ds.msi < 500).all("band").compute()
ds = ds.where(mask, drop=True)

# %% Calculate residuals
ds["residual_arts"] = get_bias(ds.sel(band="TIR2"), var_name1="arts", var_name2="msi")
ds["residual_acmcap"] = get_bias(
    ds.sel(band="TIR2"),
    var_name1="MSI_longwave_brightness_temperature_forward",
    var_name2="MSI_longwave_brightness_temperature",
)

# %% Calculate cloud top temperature at -30 dBZ
threshold_dbz = -30
ds["ctt_dbz"] = get_var_at_max_height_by_dbz(ds, threshold_dbz=threshold_dbz, var_name="temperature")

# Calculate core thickness
threshold_dbzs = [-20, -30]
ds["temp_diff_dbz_drop"] = get_difference_by_dbzs(ds, threshold_dbzs=threshold_dbzs, var_name="temperature")

# %% The 'blob' outliers
threshold_cold = 225  # K
threshold_residual = 20  # K

groups = ds.sel(band="TIR2", habit=Habit.Column, psd=PSD.D14).groupby(
    {
        "msi": xr.groupers.BinGrouper(bins=[180, threshold_cold, 300], labels=["cold_cloud", "warm_cloud"]),
        "residual_arts": xr.groupers.BinGrouper(
            bins=[-50, -threshold_residual, threshold_residual, 50],
            labels=["negative_outliers", "inliers", "positive_outliers"],
        ),
    }
)
group_dict = dict(groups)

# %% check histogram in each group
# var_name = "ctt_dbz"
# var_name = "temp_diff_dbz_drop"
var_name = "liquid_optical_depth"
plt.figure(figsize=(8, 6))
for group_name, ds_group in group_dict.items():
    if group_name == ("warm_cloud", "negative_outliers"):
        break
    ds_group[var_name].plot.hist(
        bins=50,
        alpha=0.5,
        density=True,
        label=str(group_name) + f" (N={ds_group.sizes['along_track']})",
    )
    plt.xlabel(f"{var_name}")
    plt.ylabel("Density")
    plt.legend()

# %% check histogram of geolocation in one group
group_name = ("cold_cloud", "negative_outliers")
ds_group = group_dict[group_name]

fig, ax = plt.subplots(figsize=(8, 6))
ds_group["msi"].groupby(
    longitude=xr.groupers.BinGrouper(bins=np.linspace(-180, 180, 90), labels=None),
    latitude=xr.groupers.BinGrouper(bins=np.linspace(-90, 90, 45), labels=None),
).count().plot(x="longitude_bins", y="latitude_bins", ax=ax, add_colorbar=True)
plt.title(f"2D Histogram of Longitude and Latitude in Group {group_name}")
plt.xlabel("Longitude (degrees)")
plt.ylabel("Latitude (degrees)")

# %% count number of orbits in each group
group_name = ("cold_cloud", "negative_outliers")
ds_group = group_dict[group_name].sortby("time").isel(along_track=slice(None, max_profiles_per_group := 1000, skip_profiles := 1))
groups_orbits = ds_group.groupby(orbit_frame=xr.groupers.UniqueGrouper())
group_orbit_count = groups_orbits.map(lambda ds: ds["orbit_frame"].count("along_track"))
group_orbit_count = group_orbit_count.sortby(group_orbit_count, ascending=False)

# put other psd and habit back to the ds_group
ds_group = ds_group.assign(
    ds.swap_dims({"along_track": "time"})[["arts", "residual_arts", "residual_arts_relative"]]
    .interp(time=ds_group.time.data, method="nearest")
    .swap_dims({"time": "along_track"})
)

# %% plot profiles in the group
ds_group = ds_group.assign_coords(profile_idx=("along_track", np.arange(ds_group.sizes["along_track"]))).load()

nrows = 2
fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 3 * nrows), sharex=True, constrained_layout=True)

_t, _h, _z = xr.broadcast(
    ds_group["profile_idx"],
    ds_group["height"].fillna(-1000),
    ds_group["reflectivity_corrected"],
)
ax[0].pcolormesh(_t, _h, _z, vmin=-30, vmax=30, cmap=colormaps.chiljet2)
ax[0].set_ylabel("Height (m)")
ax[0].set_xlabel("Profile Index in the Group")
ax[0].set_title(f"Profiles in Group {group_name}", fontsize="xx-large")

cbar = fig.colorbar(ax[0].collections[0], ax=ax[0], label="Reflectivity (dBZ)")

ds_group["residual_arts"].squeeze().plot(
    ax=ax[1],
    hue="psd",
    x="profile_idx",
    marker=".",
    ls="-",
    label=ds_group.psd.data,
    add_legend=False,
)
ds_group[["residual_xgb", "residual_acmcap"]].to_array().plot(
    ax=ax[1],
    hue="variable",
    x="profile_idx",
    marker="x",
    ls="--",
    label=["XGBoost", "ACMCAP"],
    add_legend=False,
)
ax[1].set_title("Residuals (True - Predicted)", fontsize="xx-large")
ax[1].set_ylabel("Residual (K)")
ax[1].set_xlabel("Profile Index in the Group")
ax[1].legend(loc="upper right")

# move legend to outside the plot
ax[1].legend(loc="upper right", frameon=False)
ax[1].set_ylim(-50, 50)
ax[1].axhline(0, color="red", linestyle="--", alpha=0.5)
ax[-1].set_xlabel("Profile Index in the Group")


# %% plot scatter plots of true vs predicted for different habits and psds, and also for ACMCAP and XGBoost
habits, psds = [Habit.Column], [PSD.D14, PSD.F07T, PSD.MDG]

ncols = len(psds)
nrows = len(habits) + 1
fig, ax = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(3 * ncols, 3 * nrows),
    sharex=True,
    sharey=True,
    constrained_layout=True,
    squeeze=False,
)
for i, habit in enumerate(habits):
    for j, psd in enumerate(psds):
        y_true = ds_group.sel(habit=habit, psd=psd)["msi"].data.flatten()
        y_pred = ds_group.sel(habit=habit, psd=psd)["arts"].data.flatten()
        ax[i, j].scatter(
            y_true,
            y_pred,
            s=5,
            alpha=0.5,
        )
        ax[i, j].set_title(f"{psd}\n{habit}")

# ACMCAP
i = nrows - 1
ax[i, 0].scatter(
    ds_group["MSI_longwave_brightness_temperature"].data.flatten(),
    ds_group["MSI_longwave_brightness_temperature_forward"].data.flatten(),
    s=5,
    alpha=0.5,
)
ax[i, 0].set_title(f"ACMCAP")

# XGBoost
ax[i, 1].scatter(
    ds_group["msi"].data.flatten(),
    ds_group["xgb"].data.flatten(),
    s=5,
    alpha=0.5,
)
ax[i, 1].set_title(f"XGBoost")

for r in range(nrows):
    for c in range(ncols):
        ax[r, c].set_xlabel("True [K]") if r == nrows - 1 else None
        ax[r, c].set_ylabel("Predicted [K]") if c == 0 else None
        ax[r, c].plot([180, 300], [180, 300], "r--", alpha=0.5)

ax[-1, -1].remove()  # turn off the last empty subplot

# %% check relative residuals in the group
ncols = len(psds)
nrows = len(habits) + 1
fig, ax = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(3 * ncols, 3 * nrows),
    sharex=True,
    sharey=True,
    constrained_layout=True,
    squeeze=False,
)
for i, habit in enumerate(habits):
    for j, psd in enumerate(psds):
        ds_group.sel(habit=habit, psd=psd)["residual_arts_relative"].plot.hist(
            bins=np.linspace(-0.3, 0.3, 60),
            yscale="log",
            ax=ax[i, j],
        )
        ax[i, j].set_title(f"{psd}\n{habit}")

# ACMCAP and XGBoost
i = nrows - 1
ds_group["residual_acmcap_relative"].plot.hist(
    ax=ax[i, 0],
    bins=np.linspace(-0.3, 0.3, 60),
    yscale="log",
)
ax[i, 0].set_title(f"ACMCAP")

# XGBoost
ds_group["residual_xgb_relative"].plot.hist(
    ax=ax[i, 1],
    bins=np.linspace(-0.3, 0.3, 60),
    yscale="log",
)
ax[i, 1].set_title(f"XGBoost")
ax[-1, -1].remove()  # turn off the last empty subplot

# %% check correlation between arts-msi and cloud top temperature
threshold_temp = 225  # K
groups_dict = dict(
    ds.sel(band="TIR2").groupby(msi=xr.groupers.BinGrouper(bins=[180, threshold_temp, 300], labels=["cold_cloud", "warm_cloud"]))
)
group_name = "cold_cloud"
habits, psds = [Habit.Column], [PSD.D14, PSD.F07T, PSD.MDG]
var = "temp_diff_dbz_drop"
bins_y = np.arange(-50, 51, 2)
# var = "ctt_dbz"
# bins_y = np.arange(190, 285, 5)

# bins_x = np.arange(-50, 51, 2)
bins_x = np.linspace(-0.3, 0.3, 60)
H = np.empty((len(habits) + 1, len(psds)), dtype=object)
n_samples = np.empty((len(habits) + 1, len(psds)))
for i, habit in enumerate(habits):
    for j, psd in enumerate(np.unique(psds)):
        x = groups_dict[group_name]["residual_arts_relative"].sel(habit=habit, psd=psd).values.flatten()
        y = groups_dict[group_name][var].values.flatten()
        n_profiles = np.sum(~np.isnan(x) & ~np.isnan(y))
        print(f"Number of valid profiles for {psd}, {habit}: {n_profiles}")
        hist, xedges, yedges = np.histogram2d(
            x=x,
            y=y,
            bins=(bins_x, bins_y),
        )
        H[i, j] = hist
        n_samples[i, j] = n_profiles

# ACMCAP
x = groups_dict[group_name]["residual_acmcap_relative"].values.flatten()
y = groups_dict[group_name][var].values.flatten()
n_profiles = np.sum(~np.isnan(x) & ~np.isnan(y))
print(f"Number of valid profiles for ACMCAP: {n_profiles}")
hist, xedges, yedges = np.histogram2d(
    x=x,
    y=y,
    bins=(bins_x, bins_y),
)
H[-1, 0] = hist
n_samples[-1, 0] = n_profiles

# XGBoost
x = groups_dict[group_name]["residual_xgb_relative"].values.flatten()
y = groups_dict[group_name][var].values.flatten()
n_profiles = np.sum(~np.isnan(x) & ~np.isnan(y))
print(f"Number of valid profiles for XGBoost: {n_profiles}")
hist, xedges, yedges = np.histogram2d(
    x=x,
    y=y,
    bins=(bins_x, bins_y),
)
H[-1, 1] = hist
n_samples[-1, 1] = n_profiles

# %% plot 2D histograms
n_row = len(habits) + 1
n_col = len(psds)
fig, ax = plt.subplots(
    n_row,
    n_col,
    figsize=(6 * n_col, 6 * n_row),
    sharex=True,
    sharey=True,
    squeeze=False,
    constrained_layout=True,
)
norm = LogNorm(vmax=np.max([h.max() for h in H.flatten()[:-1]]), vmin=1)
for i, habit in enumerate(habits):
    for j, psd in enumerate(psds):
        ax[i, j].pcolormesh(
            bins_x,
            bins_y,
            H[i, j].T,
            cmap="Blues",
            norm=norm,
        )
        ax[i, j].set_title(f"{psd}\n{habit}")
        ax[i, j].set_xlabel("(MSI - ARTS)/MSI")

# ACMCAP
i = n_row - 1
ax[i, 0].pcolormesh(
    bins_x,
    bins_y,
    H[i, 0].T,
    cmap="Blues",
    norm=norm,
)
ax[i, 0].set_title(f"ACMCAP")
ax[i, 0].set_xlabel("(MSI - ACMCAP)/MSI")

# XGBoost
i = n_row - 1
ax[i, 1].pcolormesh(
    bins_x,
    bins_y,
    H[i, 1].T,
    cmap="Blues",
    norm=norm,
)
ax[i, 1].set_title(f"XGBoost")
ax[i, 1].set_xlabel("(MSI - XGBoost)/MSI")

# fig.colorbar(
#     ax[0, 0].collections[0],
#     ax=ax,
#     orientation="vertical",
#     label="Counts",
#     aspect=40,
# )
ax[-1, -1].axis("off")  # turn off the empty subplot

for i in range(n_row):
    for j in range(n_col):
        ax[i, j].set_ylabel(f"diff[K] at {threshold_dbzs} dBZ") if j == 0 else None
        ax[i, j].grid()
        (
            ax[i, j].text(
                0.05,
                0.95,
                f"N={n_samples[i,j]}",
                transform=ax[i, j].transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            if (i, j) != (n_row - 1, n_col - 1)
            else None
        )


fig.suptitle(f"{group_name} {threshold_temp} K", fontsize="x-large")
fig.tight_layout()

# %% one orbit to visualize cloud top height and core thickness
from ectools import ecplot

orbit_frame_to_look = group_orbit_count.isel(orbit_frame=2).orbit_frame.data
# orbit_frame_to_look = orbit_frames[0]
ds_one_orbit = ds.groupby("orbit_frame")[str(orbit_frame_to_look)]
ds_one_orbit = (
    ds_one_orbit.sortby("time")
    .swap_dims({"along_track": "time"})
    .sel(time=slice("2025-07-08T09:37:00", None))
    .swap_dims({"time": "along_track"})
)

ds_one_orbit.encoding["source"] = ds_one_orbit.attrs["CPR source"]
threshold_dbzs = [-20, -30]
ds_one_orbit["temp_diff_dbz_drop"] = get_difference_by_dbzs(ds_one_orbit, threshold_dbzs=threshold_dbzs, var_name="temperature")

nrows = 3
fig, ax = plt.subplots(nrows, 1, figsize=(25, 7 * nrows), constrained_layout=True)

# reflectivity plot
ecplot.plot_EC_2D(
    ax=ax[0],
    ds=ds_one_orbit,
    varname="reflectivity_corrected",
    label="Z",
    units="dBZ",
    plot_scale="linear",
    plot_range=[-30, 30],
    use_localtime=False,
)

# cloud top height plot
# ecplot.plot_EC_1D(
#     ax=ax[0],
#     ds=ds_one_orbit,
#     plot1D_dict={
#         f"Cloud Top {dbz}": {
#             "xdata": ds_one_orbit["time"],
#             "ydata": get_var_at_max_height_by_dbz(ds_one_orbit, threshold_dbz=dbz, var_name="height"),
#             "color": "red",
#         }
#         for dbz in threshold_dbzs
#     },
#     ylabel="Cloud Top Height",
#     title="Cloud Top Height from Reflectivity Profiles",
#     include_ruler=False,
#     use_localtime=False,
# )

# residuals plot
plot1D_dict = {
    f"arts {p}": {
        "xdata": ds_one_orbit["time"],
        "ydata": ds_one_orbit["residual_arts"].sel(psd=p, habit=Habit.Column),
    }
    for p in [PSD.D14, PSD.F07T, PSD.MDG]
}
plot1D_dict.update(
    {
        "acmcap": {
            "xdata": ds_one_orbit["time"],
            "ydata": ds_one_orbit["residual_acmcap"],
        },
        "xgboost": {
            "xdata": ds_one_orbit["time"],
            "ydata": ds_one_orbit["residual_xgb"],
        },
    }
)
ecplot.plot_EC_1D(
    ax=ax[1],
    ds=ds_one_orbit,
    plot1D_dict=plot1D_dict,
    ylabel="(MSI - Predicted) [K]",
    title="Residuals",
    include_ruler=False,
    use_localtime=False,
)
ax[1].axhline(0, color="black", linestyle="--", alpha=0.5)

# temperature difference at dBZ thickness plot
# ecplot.plot_EC_1D(
#     ax=ax[2],
#     ds=ds_one_orbit,
#     plot1D_dict={
#         "temp_diff_dbz_drop": {
#             "xdata": ds_one_orbit["time"],
#             "ydata": ds_one_orbit["temp_diff_dbz_drop"],
#             "color": "black",
#         },
#     },
#     ylabel="Temp Difference at dBZ Thickness (K)",
#     title="Temp Difference at dBZ Thickness from Reflectivity Profiles",
#     include_ruler=False,
#     use_localtime=False,
# )

ecplot.plot_EC_1D(
    ax=ax[2],
    ds=ds_one_orbit,
    plot1D_dict={
        "ctt_dbz": {
            "xdata": ds_one_orbit["time"],
            "ydata": ds_one_orbit["ctt_dbz"],
            "color": "black",
        },
    },
    ylabel="Cloud Top Temperature (K)",
    title="Cloud Top Temperature from Reflectivity Profiles",
    include_ruler=False,
    use_localtime=False,
)
