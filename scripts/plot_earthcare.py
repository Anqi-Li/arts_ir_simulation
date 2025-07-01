# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binned_statistic
import glob

# %% read an input file
path = f"../data/earthcare/arts_input_data/arts_input_{'03879B'}.nc"
ds = xr.open_dataset(path)

fig, ax = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, sharex=True)
ds.dBZ.plot(ax=ax[0], x="nray", y="height_grid", vmin=-30, vmax=30, add_colorbar=True)
ds.pixel_values.plot(ax=ax[1], x="nray", marker=".", markersize=1, label="MSI")


# %% read output files
habit_std = ["LargePlateAggregate", "8-ColumnAggregate"][0]
psd = ["DelanoeEtAl14", "FieldEtAl07TR", "FieldEtAl07ML"][1]
orbit_frame = "*"  # or your specific pattern

pattern = (
    f"../data/earthcare/arts_output_data/cold_allsky_{habit_std}_{psd}_{orbit_frame}.nc"
)
matching_files = glob.glob(pattern)
print(f"Number of matching files: {len(matching_files)}")

ds_arts = xr.open_mfdataset(
    matching_files,
    chunks="auto",
    combine="nested",
    concat_dim="nray",
    parallel=True,
).sortby("profileTime")
ds_arts["height_grid"] = (
    ds_arts["height_grid"].pipe(lambda x: x * 1e-3).assign_attrs(units="km")
)  # Convert height_grid to km

ds_arts["cloud_top_T"] = (
    ds_arts["temperature"]
    .where(ds_arts["frozen_water_content"] > 1e-5)
    .min("height_grid")
)
# %%
# plot the results
fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 8), constrained_layout=True)

# plot the dBZ, FWC, and LWC profiles
kwargs = dict(
    y="height_grid",
    x="nray",
    add_colorbar=True,
)
# ds_arts["dBZ"].where(ds_arts["dBZ"] > -30).plot(ax=axes[0], vmin=-30, vmax=30, **kwargs)
ds_arts["temperature"].plot(ax=axes[0], vmin=200, vmax=300, **kwargs)
ds_arts["bulkprop"].sel(scat_species="FWC").pipe(np.log10).plot(
    ax=axes[1], vmin=-6, vmax=-3, **kwargs
)
ds_arts["bulkprop"].sel(scat_species="LWC").pipe(np.log10).plot(
    ax=axes[2], vmin=-7, vmax=-1, **kwargs
)

# for x in np.where(ds_arts.time.diff("nray").load() > pd.to_timedelta("60s"))[0]:
#     axes[0].axvline(x=x, ymin=0, ymax=1, color="r", ls="-", lw=1)

# plot the brightness temperature
kwargs = dict(x="nray", marker=".", markersize=0.3, ls="-", lw=0.3, ax=axes[3])
ds_arts["cloud_top_T"].plot(label="cloud top T", **kwargs)
ds_arts["arts"].mean("f_grid").plot(label="arts", **kwargs)
ds_arts["pixel_values"].plot(label="MSI", **kwargs)
axes[3].legend(loc="center left")

# plot the difference between ARTS and MSI brightness temperature
kwargs = dict(x="nray", marker=".", markersize=0.3, ls="-",lw=0.3, ax=axes[4], ylim=[-55, 55])
ds_arts[["pixel_values", "cloud_top_T"]].to_array().diff("variable").assign_attrs(
    units="K"
).plot(label="cloud top T", **kwargs)
ds_arts[["pixel_values", "arts"]].to_array().diff("variable").mean(
    "f_grid"
).assign_attrs(units="K").plot(label="arts", **kwargs)
axes[4].axhline(0, color="k", ls="--", lw=0.5)
axes[4].legend(loc="center left")

axes[0].set_title("Temperature")
axes[1].set_title("FWC (log10)")
axes[2].set_title("LWC (log10)")
axes[3].set_title("Brightness Temperature (mean over f_grid)")
axes[4].set_title("Diff (X - MSI)")
axes[0].set_xlabel("")
axes[1].set_xlabel("")
axes[2].set_xlabel("")
axes[3].set_xlabel("")

fig.suptitle(
    f"""{habit_std}
{psd}
{orbit_frame}
    """,
)
# put some text under the figure
plt.figtext(
    0.5,
    -0.1,
    """
    This figure shows the results of the ARTS simulation for EarthCARE data.
    The first three panels show the T, FWC (log10), and LWC (log10) profiles.
    The fourth panel shows the brightness temperature from ARTS and MSI.
    The last panel shows the difference between ARTS and MSI brightness temperature.
""",
    ha="center",
    fontsize=10,
    wrap=True,
)
plt.show()

# %% plot conditional probability density function of arts and msi brightness temperature


def plot_cond_distribution(y_true, y_pred, bin_edges=np.arange(200, 280, 2), title=""):

    # Calculate conditional probability

    bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means, _, binnumber = binned_statistic(
        y_true, y_pred, statistic="mean", bins=bin_edges
    )
    bin_per90, _, _ = binned_statistic(
        y_true, y_pred, statistic=lambda x: np.percentile(x, 90), bins=bin_edges
    )
    bin_per10, _, _ = binned_statistic(
        y_true, y_pred, statistic=lambda x: np.percentile(x, 10), bins=bin_edges
    )

    h_joint_test_pred, _, _ = np.histogram2d(
        y_true, y_pred, bins=bin_edges, density=True
    )
    h_test, _ = np.histogram(y_true, bins=bin_edges, density=True)
    h_conditional = h_joint_test_pred / h_test.reshape(-1, 1)
    h_conditional_nan = np.where(h_conditional > 0, h_conditional, np.nan)

    fig, ax = plt.subplots(
        2, 1, height_ratios=[3, 1], sharex=True, figsize=(8, 8), constrained_layout=True
    )
    ax[0].plot(bin_mid, bin_means, label="Mean", c="k")
    ax[0].plot(bin_mid, bin_per90, label="90th percentile")
    ax[0].plot(bin_mid, bin_per10, label="10th percentile")
    # c = ax[0].contourf(bin_mid, bin_mid, h_conditional_nan.T, cmap="Blues", vmin=0, vmax=0.14)
    c = ax[0].pcolormesh(
        bin_edges, bin_edges, h_conditional_nan.T, cmap="Blues", vmin=0, vmax=0.14
    )
    plt.colorbar(c, label=f"P(Predicted | True)")
    # Add diagonal line
    ax[0].plot(
        [bin_mid[0], bin_mid[-1]],
        [bin_mid[0], bin_mid[-1]],
        "r--",
        label="True = Predicted",
    )
    ax[0].legend()
    ax[0].set_xlim(bin_edges[0], bin_edges[-1])
    ax[0].set_ylim(bin_edges[0], bin_edges[-1])
    ax[0].set_ylabel("Predicted [K]")
    ax[0].set_title(title)

    ax[1].plot(bin_mid, h_test, label="", c="C0")
    ax[1].set_ylabel("P(True)")
    ax[1].set_xlabel("True [K]")
    plt.show()


plot_cond_distribution(
    y_true=ds_arts["pixel_values"].values.flatten(),
    # y_pred=ds_arts["arts"].mean("f_grid").values.flatten(),
    y_pred=ds_arts["cloud_top_T"].values.flatten(),
    title=f"""
    {habit_std}
    {psd}
    {orbit_frame}
    """,
)

# %%
ds_arts.pixel_values.plot.hist(
    bins=np.arange(200, 300, 2),
    density=True,
    label="MSI",
    alpha=0.5,
    color="C0",
)
