# %%
import os

# limit BLAS/OMP threads (do this before importing numpy/scipy/xarray if possible)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import xgboost as xgb
import numpy as np
import data_paths as dp
from sklearn.model_selection import train_test_split
from get_earthcare import load_and_merge_input_data
import xarray as xr
from glob import glob
from plotting import calculate_conditional_probabilities, plot_conditional_panel
import matplotlib.pyplot as plt
from plotting import get_ml_xy

# %%
model_tag = "second_try"

# %%
orbit_frame_list_in_august = dp.get_common_orbit_frame_list(year="2025", month="08", day="*")[:]
orbit_frame_list_in_august = sorted(orbit_frame_list_in_august)[::10]  # use every 2nd orbit for faster testing
all_aligned_file_list = glob(dp.MRGR_TIR_aligned + "/*.nc")
input_file_list = [f for f in all_aligned_file_list if f.split("_")[-1].split(".")[0] in orbit_frame_list_in_august]
print(f"Number of files to be loaded: {len(input_file_list)}")

# %% save all_file_list to a txt file
file_path = os.path.join(dp.XGBoost, f"input_file_list_{model_tag}.txt")

if save := False:
    with open(file_path, "w") as f:
        for item in input_file_list:
            f.write("%s\n" % item)

# %% read input_file_list from txt file
file_path = os.path.join(dp.XGBoost, f"input_file_list_{model_tag}.txt")
if read := False:
    with open(file_path, "r") as f:
        input_file_list = f.read().splitlines()

# %%
# use a small dask local cluster / client
from dask.distributed import Client

# e.g. 2 workers × 2 threads each (adjust to your share policy)
client = Client(
    n_workers=2,
    threads_per_worker=2,
    # memory_limit="4GB",
)
print(client)  # shows cluster config

# %%
ds = xr.open_mfdataset(
    input_file_list,
    combine="nested",
    concat_dim="along_track",
    parallel=True,
    preprocess=lambda ds: ds.assign({"orbit_frame": ds.encoding["source"].split("_")[-1].split(".")[0]}),
    # chunks={"along_track": 1000},
)

low_reflectivity_treshold = -15
skip_profiles = 5
low_reflectivity = (ds.reflectivity_corrected.fillna(-999) < low_reflectivity_treshold).all(dim="CPR_height").compute()
ds_masked = ds.where(~low_reflectivity, drop=True).isel(along_track=slice(None, None, skip_profiles))
ds_org = ds.copy(deep=True)
ds = ds_masked
print(
    f"Original dataset size: {ds_org.dims['along_track']}, \n {ds.dims['along_track']} after masking low reflectivity profiles and skipping every {skip_profiles} profiles"
)


# %% Define features and target variable
split_idx = int(0.8 * ds.dims["along_track"])
ds_train = ds.isel(along_track=slice(0, split_idx))
ds_test = ds.isel(along_track=slice(split_idx, None))
dtrain = get_ml_xy(ds_train)
dtest = get_ml_xy(ds_test)

# %% Train XGBoost model
# Set XGBoost parameters
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
    "nthread": 4,  # <-- limit CPU cores used by XGBoost
}

# Train the model
evals = [(dtrain, "train"), (dtest, "eval")]

if "model" in locals():
    print("Continuing training from existing model.")
xgb_model = locals().get("model", None)

evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=10,
    evals_result=evals_result,
    xgb_model=xgb_model,
)

# %% Save the model
if save := False:
    model.save_model(os.path.join(dp.XGBoost, f"xgboost_model_{model_tag}.json"))
    print("Model saved.")

# %% Load the model
if load := False:
    model_tag = "first_try"
    model = xgb.Booster()
    model.load_model(os.path.join(dp.XGBoost, f"xgboost_model_{model_tag}.json"))
    print("Model loaded.")

# %% prediction on test set
y_test = dtest.get_label().reshape(dtest.num_row(), 3)
y_pred = model.predict(dtest)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"Test RMSE: {rmse:.4f}")

# %%
from matplotlib.colors import LogNorm, NoNorm
from matplotlib.ticker import FuncFormatter, LogLocator, LogFormatter
from plotting import sci_formatter

bin_edges, bin_means, bin_per90, bin_per10, h_test, h_conditional_nan = calculate_conditional_probabilities(
    y_true=y_test[:, 1], y_pred=y_pred[:, 1], bin_edges=np.arange(180, 320, 5)
)

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plot_conditional_panel(
    ax=ax,
    bin_edges=bin_edges,
    bin_means=bin_means,
    bin_per90=bin_per90,
    bin_per10=bin_per10,
    h_conditional_nan=h_conditional_nan,
    title="XGBoost",
    norm=LogNorm(),
)
# annotate sample size
ax.text(0.05, 0.95, f"N={len(y_test)}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
ax.set_xlabel("True Values (K)")
ax.set_ylabel("Predicted Values (K)")
cbar = plt.colorbar(
    ax.collections[0],
    ax=ax,
    orientation="vertical",
    label="P (Predicted | True)",
    aspect=40,
)
cbar.locator = LogLocator(base=10)
cbar.update_ticks()
cbar.formatter = FuncFormatter(sci_formatter)

# ax.plot(bin_edges, bin_edges + 30, color="red", linestyle="--", label="+30K")
# ax.legend()
# %% Calculate residual and combine to original dataset
# ds_test = ds.isel(along_track=slice(-len(y_test), None))
ds_test = ds_test.assign(
    {
        "msi_predicted": (("along_track", "band"), y_pred),
        "msi_residual": (("along_track", "band"), y_test - y_pred),
        "msi_residual_percent": (("along_track", "band"), (y_test - y_pred) / y_test * 100),
    }
)

groups_by_residual = dict(
    ds_test.sel(band="TIR2").groupby(
        msi_residual=xr.groupers.BinGrouper(bins=[-50, -20, 20, 50], labels=["negative", "neutral", "positive"]),
    )
)
# %% Plot example of largest negative residual orbit
values, counts = groups_by_residual["negative"].orbit_frame.pipe(np.unique, return_counts=True)
orbit_to_check = values[np.argsort(counts)[::-1]][1]  # most frequent orbit with large negative residuals

ds_one_orbit_test = ds_test.groupby("orbit_frame")[orbit_to_check]
ds_one_orbit = ds_org.groupby("orbit_frame")[orbit_to_check].merge(ds_one_orbit_test)

fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
ds_one_orbit["msi"].sel(band="TIR2").plot(ax=ax[0], label="True MSI", color="blue")
ds_one_orbit["msi_predicted"].sel(band="TIR2").plot(ax=ax[0], marker=".", label="Predicted MSI", color="orange")
ax[0].set_title(f"Orbit Frame: {orbit_to_check} - True vs Predicted MSI")
ax[0].legend()

ds_one_orbit["msi_residual"].sel(band="TIR2").plot(ax=ax[1], marker=".", color="green")
ax[1].axhline(0, color="red", linestyle="--")
ax[1].set_title("MSI Residual (True - Predicted)")

ax[2].pcolormesh(
    ds_one_orbit["along_track"],
    ds_one_orbit["height"].fillna(0).transpose("CPR_height", "along_track"),
    ds_one_orbit["reflectivity_corrected"].transpose("CPR_height", "along_track"),
    # shading="auto",
    cmap="viridis",
)
ax[2].set_title("Reflectivity Corrected")
ax[2].set_ylabel("Height (m)")
plt.tight_layout()
# %%
