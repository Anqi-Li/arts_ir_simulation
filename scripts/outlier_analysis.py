# %%
import numpy as np
import matplotlib.pyplot as plt
from plotting import (
    plot_outliers_analysis,
    load_arts_output_data,
    plot_dBZ_IR,
    plot_dBZ_fwc_IR,
)
import cartopy.crs as ccrs

file_pattern = (
    "../data/earthcare/arts_output_data/high_fwp_5th_{habit_std}_{psd}_{orbit_frame}.nc"
)

# %%
print("Loading ARTS output data...")

# Load some example data
habit_std_idx = 0  # Change as needed
psd_idx = 0  # Change as needed

habit_std, psd, orbits, ds_arts = load_arts_output_data(
    habit_std_idx, psd_idx, n_files=10, random_seed=45, file_pattern=file_pattern
)  # Limit files for faster processing

# Extract y_true and y_pred data
y_true = ds_arts["pixel_values"].values.flatten()
y_pred = ds_arts["arts"].mean("f_grid").values.flatten()

print(f"Data loaded: {len(y_true)} data points")

print("\n" + "=" * 60)
print("ANALYSIS PLOTS")
print("=" * 60)

fig, axes, outliers = plot_outliers_analysis(
    y_true,
    y_pred,
    threshold_method="absolute",
    threshold_value=20.0,  # Change as needed
    figsize=(15, 5),
)
axes[2].set_visible(False)
fig.suptitle(f"{habit_std}\n{psd}\n{len(ds_arts.nray)} profiles", fontsize=20, x=0.35)
plt.show()

# %% Check dBZ profiles
ds_inliners_subset = ds_arts.drop_isel(nray=outliers["indices"]).isel(nray=slice(400))
ds_outliers_neg = ds_arts.isel(
    nray=outliers["indices"][np.where(outliers["residuals"] < 0)]
).isel(nray=slice(400))
ds_outliers_pos = ds_arts.isel(
    nray=outliers["indices"][np.where(outliers["residuals"] > 0)]
).isel(nray=slice(400))

plot=False
if plot:
    fig, axes = plot_dBZ_fwc_IR(ds_outliers_pos)
    fig.suptitle("Outliers Positive")
    axes[0].legend().set_visible(False)
    plt.show()

    fig, axes = plot_dBZ_fwc_IR(ds_outliers_neg)
    fig.suptitle("Outliers Negative")
    axes[0].legend().set_visible(False)
    plt.show()

    fig, axes = plot_dBZ_fwc_IR(ds_inliners_subset)
    fig.suptitle("Inliers Subset (200 samples)")
    axes[0].legend().set_visible(False)
    plt.show()


# %% check the geolocation of outliers

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add only coastlines for a cleaner look
ax.coastlines(resolution="50m", linewidth=0.8)
ax.gridlines(draw_labels=True)

# Your data
ds_outliers_pos.plot.scatter(
    x="longitude",
    y="latitude",
    c="red",
    ax=ax,
    transform=ccrs.PlateCarree(),
    s=20,
    label="Outliers Pos",
)
ds_outliers_neg.plot.scatter(
    x="longitude",
    y="latitude",
    c="blue",
    ax=ax,
    transform=ccrs.PlateCarree(),
    s=20,
    label="Outliers Neg",
)

# Set extent to show your data region
ax.set_global()  # or ax.set_extent([lon_min, lon_max, lat_min, lat_max])
ax.legend()
ax.set_title("MSI vs ARTS: Inliers and Outliers")
plt.show()

#%%
orbit_frame=['03871C', '04358B', '05093E', '04701A','05128A','03871A','04025A','04247F','04299F','04738H','05093E']
habit_std, psd, orbits, ds_arts = load_arts_output_data(
    0, 0, n_files=10, random_seed=2, file_pattern=file_pattern
)
fig, ax = plot_dBZ_fwc_IR(ds_arts.sortby('time')) 
ax[0].legend().set_visible(False)
# %%
