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
    get_var_at_max_height_by_dbz,
    get_difference_by_dbzs,
)
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

# %% Load XGBoost model and make predictions
import xgboost as xgb

model_tag = "second_try"
model = xgb.Booster()
model.load_model(os.path.join(dp.XGBoost, f"xgboost_model_{model_tag}.json"))
print("Model loaded.")
dtest = get_ml_xy(ds)
y_pred = model.predict(dtest)
ds = ds.assign({"xgb": (("along_track", "band"), y_pred)})

# %% Calculate (and save) conditional probabilities for ARTS
save = False
overwrite = False
filename_pattern_arts = "../data/earthcare/arts_output_statistics/full_conditional_probabilities_{psd}_{habit}.npy"

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

# %% Plot conditional probability for 3 channels in ACM_CAP
y_true_acmcap = ds["MSI_longwave_brightness_temperature"]
y_pred_acmcap = ds["MSI_longwave_brightness_temperature_forward"]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(3, 3 * 3), squeeze=False, constrained_layout=True)
for i, band in enumerate(["TIR1", "TIR2", "TIR3"]):
    (
        _,
        bin_means_acmcap,
        bin_per90_acmcap,
        bin_per10_acmcap,
        _,
        h_conditional_nan_acmcap,
        bin_counts_acmcap,
    ) = calculate_conditional_probabilities(
        y_true=y_true_acmcap.isel(band=i).load().data,
        y_pred=y_pred_acmcap.isel(band=i).load().data,
        bin_edges=bin_edges,
        return_sample_size=True,
    )

    plot_conditional_panel(
        ax=ax[0, i],
        bin_edges=bin_edges,
        bin_means=bin_means_acmcap,
        bin_per90=bin_per90_acmcap,
        bin_per10=bin_per10_acmcap,
        h_conditional_nan=h_conditional_nan_acmcap,
        norm=LogNorm(
            vmin=1e-5,
            vmax=2e-1,
        ),
        title=f"ACM_CAP {band}",
    )


# %% add ACMCAP and XGBoost conditional probabilities to the existing arrays
def get_y_pred_and_true(ds, title):
    if title == "ACM_CAP":
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
y_true_acmcap, y_pred_acmcap = get_y_pred_and_true(ds, "ACM_CAP")
(
    _,
    bin_means_acmcap,
    bin_per90_acmcap,
    bin_per10_acmcap,
    _,
    h_conditional_nan_acmcap,
    bin_counts_acmcap,
) = calculate_conditional_probabilities(
    y_true=y_true_acmcap,
    y_pred=y_pred_acmcap,
    bin_edges=bin_edges,
    return_sample_size=True,
)
# append to the existing arrays
bin_means_[-1, 0] = bin_means_acmcap
bin_per90_[-1, 0] = bin_per90_acmcap
bin_per10_[-1, 0] = bin_per10_acmcap
h_conditional_nan_[-1, 0] = h_conditional_nan_acmcap
bin_counts_[-1, 0] = bin_counts_acmcap

# save ACMCAP conditional probabilities
if save := False:
    filename_pattern_acmcap = "../data/earthcare/acmcap_statistics/conditional_probabilities_ACMCAP.npy"
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

y_true_xgb, y_pred_xgb = get_y_pred_and_true(ds, "XGBoost")
(
    _,
    bin_means_xgb,
    bin_per90_xgb,
    bin_per10_xgb,
    _,
    h_conditional_nan_xgb,
    bin_counts_xgb,
) = calculate_conditional_probabilities(
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
if save := False:
    filename_pattern_xgb = "../data/earthcare/XGBoost/conditional_probabilities_XGBoost.npy"
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
bin_edges_, bin_means_, bin_per90_, bin_per10_, h_conditional_nan_, bin_counts_ = [
    np.empty((len(habits), len(psds)), dtype=object) for _ in range(6)
]
bin_edges = np.arange(180, 300, 5)
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
                f"/home/anqil/arts_ir_simulation/data/earthcare/arts_output_statistics/full_conditional_probabilities_{psd}_{habit}.npy",
                allow_pickle=True,
            ).item()
            bin_edges_[i, j] = stats_dict["bin_edges"]
            bin_means_[i, j] = stats_dict["bin_means"]
            bin_per90_[i, j] = stats_dict["bin_per90"]
            bin_per10_[i, j] = stats_dict["bin_per10"]
            h_conditional_nan_[i, j] = stats_dict["h_conditional_nan"]
            bin_counts_[i, j] = stats_dict["bin_counts"]

# %%
if load_ACMCAP_statistics := False:
    if bin_per90_.shape[0] == 1:
        extra = np.empty((1, bin_per90_.shape[1]), dtype=object)
        extra[:] = None
        bin_edges_ = np.concatenate([bin_edges_, extra], axis=0)
        bin_means_ = np.concatenate([bin_means_, extra], axis=0)
        bin_per90_ = np.concatenate([bin_per90_, extra], axis=0)
        bin_per10_ = np.concatenate([bin_per10_, extra], axis=0)
        h_conditional_nan_ = np.concatenate([h_conditional_nan_, extra], axis=0)
        bin_counts_ = np.concatenate([bin_counts_, extra], axis=0)

    stats_dict_acmcap = np.load(
        "../data/earthcare/acmcap_statistics/conditional_probabilities_ACMCAP.npy",
        allow_pickle=True,
    ).item()
    bin_edges_[-1, 0] = stats_dict_acmcap["bin_edges"]
    bin_means_[-1, 0] = stats_dict_acmcap["bin_means"]
    bin_per90_[-1, 0] = stats_dict_acmcap["bin_per90"]
    bin_per10_[-1, 0] = stats_dict_acmcap["bin_per10"]
    h_conditional_nan_[-1, 0] = stats_dict_acmcap["h_conditional_nan"]
    bin_counts_[-1, 0] = stats_dict_acmcap["bin_counts"]

if load_XGBoost_statistics := False:
    if bin_per90_.shape[0] == 1:
        extra = np.empty((1, bin_per90_.shape[1]), dtype=object)
        extra[:] = None
        bin_edges_ = np.concatenate([bin_edges_, extra], axis=0)
        bin_means_ = np.concatenate([bin_means_, extra], axis=0)
        bin_per90_ = np.concatenate([bin_per90_, extra], axis=0)
        bin_per10_ = np.concatenate([bin_per10_, extra], axis=0)
        h_conditional_nan_ = np.concatenate([h_conditional_nan_, extra], axis=0)
        bin_counts_ = np.concatenate([bin_counts_, extra], axis=0)

    stats_dict_xgb = np.load(
        "../data/earthcare/XGBoost/conditional_probabilities_XGBoost.npy",
        allow_pickle=True,
    ).item()
    bin_edges_[-1, 1] = stats_dict_xgb["bin_edges"]
    bin_means_[-1, 1] = stats_dict_xgb["bin_means"]
    bin_per90_[-1, 1] = stats_dict_xgb["bin_per90"]
    bin_per10_[-1, 1] = stats_dict_xgb["bin_per10"]
    h_conditional_nan_[-1, 1] = stats_dict_xgb["h_conditional_nan"]
    bin_counts_[-1, 1] = stats_dict_xgb["bin_counts"]

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
for j, title in enumerate(["ACM_CAP", "XGBoost"]):
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
