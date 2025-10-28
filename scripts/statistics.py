# %%
# %load_ext autoreload
# %autoreload 2
from plotting import (
    plot_conditional_panel,
    sci_formatter,
)

from analysis import (
    load_and_merge_ec_data,
    load_arts_results,
    load_and_merge_acmcap_data,
    calculate_conditional_probabilities,
    get_bias,
    get_ml_xy,
)
from shapely import get_y
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from earthcare_ir import Habit, PSD
from matplotlib.colors import LogNorm, NoNorm
from matplotlib.ticker import FuncFormatter, LogLocator, LogFormatter
import data_paths as dp
import os
from earthcare_ir import PSD, Habit
from ectools import colormaps
from ectools import ecio

# ignore invalid value warnings
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide *")
save = False

# from dask.distributed import Client

# client = Client(n_workers=2, threads_per_worker=2)
# print(client)  # shows cluster config


# %% print how many arts output files are there for different habits and psd
def get_arts_output_files(orbit_frame, habit, psd):
    path_pattern = os.path.join(dp.arts_output_TIR2, f"arts_TIR2_{orbit_frame}_{habit}_{psd}.nc")
    return glob.glob(path_pattern)


arts_output_file_count = {
    (habit, psd): len(get_arts_output_files("*", habit, psd))
    for habit in [Habit.Bullet, Habit.Column, Habit.Plate]
    for psd in [PSD.MDG, PSD.F07T, PSD.D14]
}

arts_output_file_count

# %% print how many aligned EC data files are there
path_aligned_ec_data = glob.glob(os.path.join(dp.MRGR_TIR_aligned, "*.nc"))
print("Number of aligned EC data files:", len(path_aligned_ec_data))


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
    product_baseline='BA',
    nested_directory_structure=True,
    )
orbit_frames_acmcap = [f.split("/")[-1].split("_")[-1].split(".")[0] for f in filelist_acmcap]
overlap_orbit_frames = set(orbit_frames_arts) & set(orbit_frames_acmcap)
print(f"Number of overlapping orbit frames between ARTS and ACMCAP: {len(overlap_orbit_frames)}")

orbit_frames = sorted(list(overlap_orbit_frames))

if shorten_to_test := True:
    print("Shorten the list of orbit frames for testing...")
    orbit_frames = orbit_frames[0:5]  # for testing purpose
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

# %% Load XGBoost model and make predictions
import xgboost as xgb

model_tag = "second_try"
model = xgb.Booster()
model.load_model(os.path.join(dp.XGBoost, f"xgboost_model_{model_tag}.json"))
print("Model loaded.")
dtest = get_ml_xy(ds)
y_pred = model.predict(dtest)
ds = ds.assign({"xgb": (("along_track", "band"), y_pred)})

# %% Calculate residuals
ds["residual_arts"] = get_bias(ds.sel(band="TIR2"), var_name1="msi", var_name2="arts")
ds["residual_xgb"] = get_bias(ds.sel(band="TIR2"), var_name1="msi", var_name2="xgb")
ds["residual_acmcap"] = get_bias(
    ds.sel(band="TIR2"),
    var_name1="MSI_longwave_brightness_temperature",
    var_name2="MSI_longwave_brightness_temperature_forward",
)

ds["residual_arts_relative"] = ds["residual_arts"] / ds["msi"].sel(band="TIR2")
ds["residual_xgb_relative"] = ds["residual_xgb"] / ds["msi"].sel(band="TIR2")
ds["residual_acmcap_relative"] = ds["residual_acmcap"] / ds["MSI_longwave_brightness_temperature"].sel(band="TIR2")


# %% calculate (and save) conditional probabilities for CAM_CAP data
def get_y_pred_and_true(ds, title):
    if title == "CAM_CAP":
        y_true = ds.sel(band="TIR2")["MSI_longwave_brightness_temperature"].load().data.flatten()
        y_pred = ds.sel(band="TIR2")["MSI_longwave_brightness_temperature_forward"].load().data.flatten()

    elif title == "XGBoost":
        y_true = ds.sel(band="TIR2")["msi"].load().data.flatten()
        y_pred = ds.sel(band="TIR2")["xgb"].load().data.flatten()

    return y_true, y_pred


def calculate_and_plot_conditional_metrics(ax, y_true, y_pred, bin_edges, title, norm=None, save=False):
    _, bin_means, bin_per90, bin_per10, _, h_conditional_nan, bin_counts = calculate_conditional_probabilities(
        y_true=y_true,
        y_pred=y_pred,
        bin_edges=bin_edges,
        return_sample_size=True,
    )

    if save:
        filename_name = os.path.join("../data/earthcare/acmcap_statistics", f"conditional_probabilities_{title}.npy")
        np.save(
            filename_name,
            {
                "bin_edges": bin_edges,
                "bin_means": bin_means,
                "bin_per90": bin_per90,
                "bin_per10": bin_per10,
                "h_conditional_nan": h_conditional_nan,
                "bin_counts": bin_counts,
            },
        )
        print(f"File is saved to {filename_name}")

    #  plot conditional probabilities
    if norm == None:
        norm = LogNorm(
            vmin=max(1e-6, np.nanmin(h_conditional_nan)),
            vmax=np.nanmax(h_conditional_nan),
        )
    plot_conditional_panel(
        ax=ax,
        bin_edges=bin_edges,
        bin_means=bin_means,
        bin_per90=bin_per90,
        bin_per10=bin_per10,
        h_conditional_nan=h_conditional_nan,
        norm=norm,
        title=title,
    )
    ax.set_ylabel("Predicted [K]")
    ax.set_xlabel("True [K]")

    ax.text(
        0.05,
        0.95,
        f"N={bin_counts.sum()}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
    )
    return


fig, ax = plt.subplots(1, 2, figsize=(3 * 2, 3), sharex=True, sharey=True, constrained_layout=True)

bin_edges = np.arange(180, 300, 5)
for i, title in enumerate(["XGBoost", "CAM_CAP"]):
    y_true, y_pred = get_y_pred_and_true(ds, title)
    calculate_and_plot_conditional_metrics(ax[i], y_true, y_pred, bin_edges, title, save=False, norm=LogNorm(vmin=2e-5, vmax=2e-1))

cbar = plt.colorbar(
    ax[0].collections[0],
    ax=ax[1],
    orientation="vertical",
    label="P (Predicted | True)",
    aspect=40,
)

# %% Calculate (and save) conditional probabilities for ARTS
save = False
overwrite = False
filename_pattern_arts = "../data/earthcare/arts_output_statistics/new_conditional_probabilities_{psd}_{habit}.npy"

habits, psds = [Habit.Column], [PSD.D14, PSD.F07T, PSD.MDG]
bin_edges_, bin_means_, bin_per90_, bin_per10_, h_conditional_nan_, bin_counts_ = [
    np.empty((len(habits), len(psds)), dtype=object) for _ in range(6)
]
bin_edges = np.arange(180, 300, 5)
for i, habit in enumerate(habits):
    for j, psd in enumerate(psds):
        print(f"Calculating conditional probabilities for {psd}, {habit}...")
        if save and os.path.exists(filename_pattern_arts.format(psd=psd, habit=habit)):
            if not overwrite:
                print(f"File already exists, skipping.")
                continue
            else:
                print(f"Overwriting existing file.")
        ds_sel = ds.sel(band="TIR2", psd=psd, habit=habit)[["msi", "arts"]].load()
        _, bin_means, bin_per90, bin_per10, _, h_conditional_nan, bin_counts = calculate_conditional_probabilities(
            y_true=ds_sel["msi"].data.flatten(),
            y_pred=ds_sel["arts"].data.flatten(),
            bin_edges=bin_edges,
            return_sample_size=True,
        )
        bin_edges_[i, j] = bin_edges
        bin_means_[i, j] = bin_means
        bin_per90_[i, j] = bin_per90
        bin_per10_[i, j] = bin_per10
        h_conditional_nan_[i, j] = h_conditional_nan
        bin_counts_[i, j] = bin_counts

        if save:
            np.save(
                filename_pattern_arts.format(psd=psd, habit=habit),
                {
                    "bin_edges": bin_edges,
                    "bin_means": bin_means,
                    "bin_per90": bin_per90,
                    "bin_per10": bin_per10,
                    "h_conditional_nan": h_conditional_nan,
                    "bin_counts": bin_counts,
                },
            )
            print(f"File is saved to {filename_pattern_arts.format(psd=psd, habit=habit)}")


# %% add CAMCAP and XGBoost conditional probabilities to the existing arrays
def get_y_pred_and_true(ds, title):
    if title == "CAM_CAP":
        y_true = ds.sel(band="TIR2")["MSI_longwave_brightness_temperature"].load().data.flatten()
        y_pred = ds.sel(band="TIR2")["MSI_longwave_brightness_temperature_forward"].load().data.flatten()

    elif title == "XGBoost":
        y_true = ds.sel(band="TIR2")["msi"].load().data.flatten()
        y_pred = ds.sel(band="TIR2")["xgb"].load().data.flatten()

    return y_true, y_pred


if bin_per90_.shape[0] == 1:
    extra = np.empty((1, bin_per90_.shape[1]), dtype=object)
    extra[:] = None
    bin_edges_ = np.concatenate([bin_edges_, extra], axis=0)
    bin_means_ = np.concatenate([bin_means_, extra], axis=0)
    bin_per90_ = np.concatenate([bin_per90_, extra], axis=0)
    bin_per10_ = np.concatenate([bin_per10_, extra], axis=0)
    h_conditional_nan_ = np.concatenate([h_conditional_nan_, extra], axis=0)
    bin_counts_ = np.concatenate([bin_counts_, extra], axis=0)

# ACMCAP conditional probabilities
filename_pattern_acmcap = "../data/earthcare/acmcap_statistics/conditional_probabilities_ACMCAP.npy"
y_true_acmcap, y_pred_acmcap = get_y_pred_and_true(ds, "CAM_CAP")
_, bin_means_acmcap, bin_per90_acmcap, bin_per10_acmcap, _, h_conditional_nan_acmcap, bin_counts_acmcap = (
    calculate_conditional_probabilities(
        y_true=y_true_acmcap,
        y_pred=y_pred_acmcap,
        bin_edges=bin_edges,
        return_sample_size=True,
    )
)
# append to the existing arrays
bin_means_[-1, 0] = bin_means_acmcap
bin_per90_[-1, 0] = bin_per90_acmcap
bin_per10_[-1, 0] = bin_per10_acmcap
h_conditional_nan_[-1, 0] = h_conditional_nan_acmcap
bin_counts_[-1, 0] = bin_counts_acmcap

# save ACMCAP conditional probabilities
if save:
    np.save(
        filename_pattern_acmcap,
        {
            "bin_edges": bin_edges_,
            "bin_means": bin_means_acmcap,
            "bin_per90": bin_per90_acmcap,
            "bin_per10": bin_per10_acmcap,
            "h_conditional_nan": h_conditional_nan_acmcap,
            "bin_counts": bin_counts_acmcap,
        },
    )
    print(f"File is saved to {filename_pattern_acmcap}")

# XGBoost conditional probabilities
filename_pattern_xgb = "../data/earthcare/xgboost_statistics/conditional_probabilities_XGBoost.npy"
y_true_xgb, y_pred_xgb = get_y_pred_and_true(ds, "XGBoost")
_, bin_means_xgb, bin_per90_xgb, bin_per10_xgb, _, h_conditional_nan_xgb, bin_counts_xgb = calculate_conditional_probabilities(
    y_true=y_true_xgb,
    y_pred=y_pred_xgb,
    bin_edges=bin_edges,
    return_sample_size=True,
)
# append to the existing arrays
bin_means_[-1, 1] = bin_means_xgb
bin_per90_[-1, 1] = bin_per90_xgb
bin_per10_[-1, 1] = bin_per10_xgb
h_conditional_nan_[-1, 1] = h_conditional_nan_xgb
bin_counts_[-1, 1] = bin_counts_xgb

# save XGBoost conditional probabilities
if save:
    np.save(
        filename_pattern_xgb,
        {
            "bin_edges": bin_edges_,
            "bin_means": bin_means_xgb,
            "bin_per90": bin_per90_xgb,
            "bin_per10": bin_per10_xgb,
            "h_conditional_nan": h_conditional_nan_xgb,
            "bin_counts": bin_counts_xgb,
        },
    )
    print(f"File is saved to {filename_pattern_xgb}")
# %% load all pre caculated statistics (conditional probabilities)
if load_statistics := False:
    habits, psds = [Habit.Column], [PSD.D14, PSD.F07T, PSD.MDG]
    np.empty((len(habits), len(psds)), dtype=object)
    bin_edges_, bin_means_, bin_per90_, bin_per10_, h_conditional_nan_ = [
        np.empty((len(habits), len(psds)), dtype=object) for _ in range(5)
    ]
    for i, habit in enumerate(habits):
        for j, psd in enumerate(psds):
            print("loading", psd, habit)

            stats_dict = np.load(
                f"/home/anqil/arts_ir_simulation/data/earthcare/arts_output_statistics/new_conditional_probabilities_{psd}_{habit}.npy",
                allow_pickle=True,
            ).item()
            bin_edges_[i, j] = stats_dict["bin_edges"]
            bin_means_[i, j] = stats_dict["bin_means"]
            bin_per90_[i, j] = stats_dict["bin_per90"]
            bin_per10_[i, j] = stats_dict["bin_per10"]
            h_conditional_nan_[i, j] = stats_dict["h_conditional_nan"]
            bin_counts_[i, j] = stats_dict["bin_counts"]


# %% plot the statistics of all psd and habit combinations
norm = LogNorm(
    vmin=max(1e-6, np.nanmin([np.nanmin(h) for h in h_conditional_nan_.flatten()[:-1]])),
    vmax=np.nanmax([np.nanmax(h) for h in h_conditional_nan_.flatten()[:-1]]),
)
# plot all panels
n_row = len(habits) + 1  # +1 for ACMCAP and XGBoost
n_col = len(psds)
fig, ax = plt.subplots(
    n_row,
    n_col,
    figsize=(3 * n_col, 3 * n_row),
    sharex=True,
    sharey=True,
    constrained_layout=True,
    squeeze=False,
)
for i in range(len(habits)):
    for j in range(len(psds)):
        plot_conditional_panel(
            ax=ax[i, j],
            bin_edges=bin_edges_[i, j],
            bin_means=bin_means_[i, j],
            bin_per90=bin_per90_[i, j],
            bin_per10=bin_per10_[i, j],
            h_conditional_nan=h_conditional_nan_[i, j],
            norm=norm,
            title=f"{psds[j]}\n{habits[i]}",
        )
        ax[i, j].text(
            0.05,
            0.95,
            f"N={bin_counts_[i, j].sum()}",
            transform=ax[i, j].transAxes,
            fontsize=10,
            verticalalignment="top",
        )

# add ACMCAP and XGBoost histograms
for j, title in enumerate(["CAM_CAP", "XGBoost"]):
    i = n_row - 1
    plot_conditional_panel(
        ax=ax[i, j],
        bin_edges=bin_edges,
        bin_means=bin_means_[-1, j],
        bin_per90=bin_per90_[-1, j],
        bin_per10=bin_per10_[-1, j],
        h_conditional_nan=h_conditional_nan_[-1, j],
        norm=norm,
        title=title,
    )
    ax[i, j].text(
        0.05,
        0.95,
        f"N={bin_counts_[-1, j].sum()}",
        transform=ax[i, j].transAxes,
        fontsize=10,
        verticalalignment="top",
    )

for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        ax[i, j].set_ylabel("Predicted [K]") if j == 0 else None
        ax[i, j].set_xlabel("True [K]") if i == ax.shape[0] - 1 else None

cbar = plt.colorbar(
    ax[0, 0].collections[0],
    ax=ax,
    orientation="vertical",
    label="P (Predicted | True)",
    aspect=40,
)
cbar.locator = LogLocator(base=10)
cbar.update_ticks()
cbar.formatter = FuncFormatter(sci_formatter)

if show_thresholds := True:
    # show outlier thresholds
    threshold_residual = 20  # K
    for i in range(len(ax.flatten())):
        ax.flatten()[i].plot(bin_edges, bin_edges + threshold_residual, "r--", alpha=0.5)
        ax.flatten()[i].plot(bin_edges, bin_edges - threshold_residual, "r--", alpha=0.5)

if show_warm_cold_line := True:
    threshold_temp = 225  # K
    for i in range(len(ax.flatten())):
        ax.flatten()[i].axvline(threshold_temp, color="green", linestyle="--", alpha=0.5)

ax[-1, -1].remove()  # turn off the last empty subplot

# fig.tight_layout()

# %% The 'blob' outliers
threshold_cold = 225  # K
threshold_residual = 20  # K

groups = ds.sel(band="TIR2", habit=Habit.Column, psd=PSD.D14).groupby(
    {
        "msi": xr.groupers.BinGrouper(bins=[180, threshold_cold, 300], labels=["cold_cloud", "warm_cloud"]),
        "residual_arts": xr.groupers.BinGrouper(
            bins=[-50, -threshold_residual, threshold_residual, 50], labels=["negative_outliers", "inliers", "positive_outliers"]
        ),
    }
)
group_dict = dict(groups)

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

_t, _h, _z = xr.broadcast(ds_group["profile_idx"], ds_group["height"].fillna(-1000), ds_group["reflectivity_corrected"])
ax[0].pcolormesh(_t, _h, _z, vmin=-30, vmax=30, cmap=colormaps.chiljet2)
ax[0].set_ylabel("Height (m)")
ax[0].set_xlabel("Profile Index in the Group")
ax[0].set_title(f"Profiles in Group {group_name}", fontsize="xx-large")

cbar = fig.colorbar(ax[0].collections[0], ax=ax[0], label="Reflectivity (dBZ)")

ds_group["residual_arts"].squeeze().plot(
    ax=ax[1], hue="psd", x="profile_idx", marker=".", ls="-", label=ds_group.psd.data, add_legend=False
)
ds_group[["residual_xgb", "residual_acmcap"]].to_array().plot(
    ax=ax[1], hue="variable", x="profile_idx", marker="x", ls="--", label=["XGBoost", "ACMCAP"], add_legend=False
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
ncols = len(psds)
nrows = len(habits) + 1
fig, ax = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows), sharex=True, sharey=True, constrained_layout=True, squeeze=False
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
    nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows), sharex=True, sharey=True, constrained_layout=True, squeeze=False
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
from plotting import get_var_at_max_height_by_dbz, get_difference_by_dbzs

threshold_dbz = -30
var = "ctt_dbz"
ds[var] = get_var_at_max_height_by_dbz(ds, threshold_dbz=threshold_dbz, var_name="temperature")
bins_y = np.arange(190, 285, 5)

# check correlation between arts-msi and core thickness
threshold_dbzs = [-20, -30]
var = "temp_diff_dbz_drop"
ds[var] = get_difference_by_dbzs(ds, threshold_dbzs=threshold_dbzs, var_name="temperature")

bins_y = np.arange(-50, 51, 2)

# %%
threshold_temp = 225  # K
groups_dict = dict(
    ds.sel(band="TIR2").groupby(msi=xr.groupers.BinGrouper(bins=[180, threshold_temp, 300], labels=["cold_cloud", "warm_cloud"]))
)
group_name = "cold_cloud"
habits, psds = [Habit.Column], [PSD.D14, PSD.F07T, PSD.MDG]
var = "temp_diff_dbz_drop"
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
# %% load all habit and/or psd arts data
# file_pattern = (
#     "../data/earthcare/arts_output_data/high_fwp_5th_{habit_std}_{psd}_{orbit_frame}.nc"
# )
# data = []
# for j in range(len(psd_list)):
#     for i in range(len(habit_std_list)):
#         data.append(
#             load_arts_output_data(
#                 i,
#                 j,
#                 file_pattern=file_pattern,
#                 n_files=100,  # test on small amount first
#             )
#         )

# %% plot FWC distribution
# fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
# ls = ["-", "--", ":"]
# c = ["#0071B2FF", "#D55C00FF", "#009E74FF"]
# for j in range(len(psd_list)):
#     for i in range(len(habit_std_list)):
#         psd = psd_list[j]
#         habit_std = habit_std_list[i]
#         ds_arts = [d[-1] for d in data if d[0] == habit_std and d[1] == psd][0]

#         bins_fwc = np.logspace(-6, -2, 50)
#         hist_fwc, _ = np.histogram(
#             ds_arts.frozen_water_content,
#             bins=bins_fwc,
#             density=True,
#         )
#         ax.plot(
#             bins_fwc[:-1],
#             hist_fwc,
#             label=f"{psd}",
#             ls=ls[i],
#             color=c[j],
#             alpha=0.8,
#         )

# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_ylim([1e3, 1e5])
# ax.set_xlabel("Frozen Water Content [kg/m³]")
# ax.set_ylabel("Probability Density")
# # ax.legend(loc="lower left", framealpha=0.3)
# ax.set_title("FWC Distribution")

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(
#     np.array(handles)[::3], np.array(labels)[::3], loc="lower left", framealpha=0.3
# )

# if save:
#     plt.savefig(
#         f"../data/figures/arts_output_fwc_distribution.png",
#         dpi=1000,
#         bbox_inches="tight",
#     )
#     print(f'Figure is saved to "../data/figures/arts_output_fwc_distribution.png"')
