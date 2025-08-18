# %%
import numpy as np
import matplotlib.pyplot as plt
from plotting import (
    plot_outliers_analysis,
    load_arts_output_data,
    plot_dBZ_IR,
)
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %%
print("Loading ARTS output data...")

# Load some example data
habit_std_idx = 0  # Change as needed
psd_idx = 2  # Change as needed

habit_std, psd, orbits, ds_arts = load_arts_output_data(
    habit_std_idx, psd_idx, n_files=80
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
ds_inliners_subset = ds_arts.drop_isel(nray=outliers["indices"]).isel(nray=slice(200))
ds_outliers_neg = ds_arts.isel(
    nray=outliers["indices"][np.where(outliers["residuals"] < 0)]
).isel(nray=slice(200))
ds_outliers_pos = ds_arts.isel(
    nray=outliers["indices"][np.where(outliers["residuals"] > 0)]
).isel(nray=slice(200))

fig, axes = plot_dBZ_IR(ds_outliers_pos)
fig.suptitle("Outliers Positive")
axes[0].legend().set_visible(False)
plt.show()

fig, axes = plot_dBZ_IR(ds_outliers_neg)
fig.suptitle("Outliers Negative")
axes[0].legend().set_visible(False)
plt.show()

fig, axes = plot_dBZ_IR(ds_inliners_subset)
fig.suptitle("Inliers Subset (200 samples)")
axes[0].legend().set_visible(False)
plt.show()

#%%
ds_outliers_neg_peak = ds_arts.isel(
    nray=outliers["indices"][np.where((outliers["residuals"] < 0) & (outliers["y_true_outliers"] < 220))]
)
fig, axes = plot_dBZ_IR(ds_outliers_neg_peak)
fig.suptitle("Outliers Negative Peak")
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

# %%

# # Example 1: Absolute threshold method
# print("\n1. ABSOLUTE THRESHOLD METHOD")
# print("-" * 40)

# outliers_abs = extract_outliers_from_diagonal(
#     y_true,
#     y_pred,
#     threshold_method="absolute",
#     threshold_value=10.0,  # Points with |residual| > 10 K
#     return_indices=True,
# )

# print(
#     f"Outliers found: {outliers_abs['n_outliers']} ({outliers_abs['outlier_fraction']:.1%})"
# )
# print(f"Threshold used: {outliers_abs['threshold_used']:.1f} K")
# print(f"Max absolute residual: {np.max(outliers_abs['abs_residuals']):.1f} K")
# print(
#     f"Mean absolute residual of outliers: {np.mean(outliers_abs['abs_residuals']):.1f} K"
# )

# # Example 2: Percentile method
# print("\n2. PERCENTILE METHOD")
# print("-" * 40)

# outliers_pct = extract_outliers_from_diagonal(
#     y_true,
#     y_pred,
#     threshold_method="percentile",
#     percentile_threshold=95,  # Top 5% of residuals
#     return_indices=True,
# )

# print(
#     f"Outliers found: {outliers_pct['n_outliers']} ({outliers_pct['outlier_fraction']:.1%})"
# )
# print(f"Threshold used: {outliers_pct['threshold_used']:.1f} K")
# print(f"Max absolute residual: {np.max(outliers_pct['abs_residuals']):.1f} K")

# # Example 3: Standard deviation method
# print("\n3. STANDARD DEVIATION METHOD")
# print("-" * 40)

# outliers_std = extract_outliers_from_diagonal(
#     y_true,
#     y_pred,
#     threshold_method="std",
#     threshold_value=2.0,  # 2 standard deviations
#     return_indices=True,
# )

# print(
#     f"Outliers found: {outliers_std['n_outliers']} ({outliers_std['outlier_fraction']:.1%})"
# )
# print(f"Threshold used: {outliers_std['threshold_used']:.1f} K")
# print(f"Max absolute residual: {np.max(outliers_std['abs_residuals']):.1f} K")

# %%
if __name__ == "__main__":
    # Run the example analysis

    # Optionally save the outliers for further analysis
    # save_outliers_to_file(outliers, "example_outliers.npz")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nNext steps you might want to try:")
    print("1. Adjust threshold values to capture different types of outliers")
    print("2. Investigate specific temperature regimes or atmospheric conditions")
    print("3. Look at spatial/temporal patterns in the outlier indices")
    print("4. Compare outliers across different habits or PSDs")
    print("5. Use the outlier indices to examine corresponding atmospheric profiles")
