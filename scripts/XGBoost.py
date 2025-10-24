# %%
import os
import data
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import xgboost as xgb
import numpy as np
import pandas as pd
import data_paths as dp
from ectools import ecio
from sklearn.model_selection import train_test_split
from get_earthcare import load_and_merge_input_data
import xarray as xr
from glob import glob
from plotting import calculate_conditional_probabilities, plot_conditional_panel
import matplotlib.pyplot as plt

# %%
model_tab = "first_try"

# %%
orbit_frame_list = dp.get_common_orbit_frame_list(year="2025", month="08", day="*")[:]

# %%
data_list = []
for orbit_frame in orbit_frame_list:
    # print(f"Processing orbit frame: {orbit_frame}")
    ds_one_frame = load_and_merge_input_data(frame_code=orbit_frame, cfmr_vars=["reflectivity_corrected"], get_xmet_temperature=True)
    data_list.append(ds_one_frame)

ds = xr.concat(data_list, dim="along_track")

# %%
all_file_list = glob(dp.MRGR_TIR_aligned + "/*.nc")
input_file_list = [f for f in all_file_list if f.split("/")[-1].split("_")[-1].split(".")[0] in orbit_frame_list]
print(f"Number of files to be loaded: {len(input_file_list)}")
# %% save all_file_list to a txt file
file_path = os.path.join(dp.XGBoost, f"input_file_list_{model_tab}.txt")

with open(file_path, "w") as f:
    for item in input_file_list:
        f.write("%s\n" % item)

# %%
ds = xr.open_mfdataset(
    input_file_list,
    combine="nested",
    concat_dim="along_track",
    parallel=True,
    # chunks={"along_track": 1000},
)
# %%
# Define features and target variable
feature_cols = ["reflectivity_corrected", "temperature", "height"]
X = ds.reset_coords()[feature_cols].transpose("along_track", "CPR_height").to_array().stack(features=["variable", "CPR_height"]).load().data
y = ds.reset_coords()["msi"].transpose("along_track", "band").load().data

mask = (np.isfinite(y).all(axis=1) if y.ndim > 1 else np.isfinite(y)) & (y < 500).all(axis=1)
X = X[mask]
y = y[mask]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# %%
# Set XGBoost parameters
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
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

# %%
# Save the model
model.save_model(os.path.join(dp.XGBoost, f"xgboost_model_{model_tab}.json"))
print("Model saved.")

# %% prediction on test set
y_pred = model.predict(dtest)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"Test RMSE: {rmse:.4f}")

# %% plot true vs predicted
plt.figure(figsize=(10, 5))
# plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
plt.hist2d(y_test[:, 1], y_pred[:, 1], bins=np.arange(180, 320, 5), norm=LogNorm(), cmap="Blues")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs Predicted Values")
plt.show()


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


# %%
