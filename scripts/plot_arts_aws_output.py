# %%
import matplotlib.colors as colors
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotting import load_arts_output_data

file_pattern = "../data/earthcare/arts_output_aws_data/aws_2nd_{habit_std}_{psd}_{orbit_frame}.nc"

# %%
habit, psd, orbits, ds_arts = load_arts_output_data(
    0,
    1,
    file_pattern=file_pattern,
    # n_files=80,  # Limit files for faster processing
)

# %% Overall AWS pdf at 4 channels
bins = np.linspace(90, 310, 100)
hist_data = []
fig, ax = plt.subplots(1, 1, figsize=(5, 6), sharex=True, sharey=True)
for i in range(1, 5):
    h, _, _ = ds_arts.sel(channel=f"AWS4{i}").arts.plot.hist(
        ax=ax,
        bins=bins,
        density=True,
        yscale="log",
        histtype="step",
        label=f"AWS4{i}",
        lw=0.8,
    )
    hist_data.append(h)
plt.ylim(1e-9, 1e-1)
plt.legend()
plt.title(f"PDFs of simulated AWS \n{habit} \n{psd} \n{len(ds_arts.nray)} samples")

# %% Simulated AWS vs MSI 2d histograms
fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
cs = None
bins_aws = np.linspace(90, 310, 100)
bins_msi = np.linspace(170, 330, 50)
# percentile levels
percentiles = [10, 25, 50, 75, 90, 95, 99]

hist2d_data = []
for i in range(1, 5):
    mask = ds_arts.sel(channel=f"AWS4{i}").to_array().isnull().any("variable").compute()
    ds_channel = ds_arts.sel(channel=f"AWS4{i}").where(~mask, drop=True)

    # Get the data
    x_data = ds_channel.arts.values
    y_data = ds_channel.pixel_values.values

    # Remove NaN values
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]

    # Create 2D histogram data first
    hist2d, xedges, yedges = np.histogram2d(x_data, y_data, bins=(bins_aws, bins_msi), density=True)
    hist2d_data.append(hist2d)

    # Create meshgrid for contour plot
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    # Compute percentiles for each histogram
    h = hist2d.T
    levels = [np.percentile(h[h > 0], p) for p in percentiles]

    # Create line contour plot with log normalization
    cs = ax.ravel()[i - 1].contour(
        X,
        Y,
        h,
        levels=levels,
        cmap="viridis",
        linewidths=1,
    )

    ax.ravel()[i - 1].set_xlabel("AWS TB [K]")
    ax.ravel()[i - 1].set_ylabel("MSI TB [K]")
    ax.ravel()[i - 1].set_title(f"AWS4{i}")

plt.tight_layout()
# Add colorbar with ticks at the contour levels and label them as percentiles
cbar = fig.colorbar(
    cs,
    ax=ax.ravel().tolist(),
    shrink=0.8,
    aspect=20,
    ticks=cs.levels,  # place ticks at the contour levels
)
# Map each level to its percentile label; assumes `percentiles` used to compute `levels`
cbar.set_ticklabels([f"{p}%" for p in percentiles])
cbar.set_label("Percentile of non-zero density")
fig.suptitle(
    f"2D Histograms of AWS(ARTS) vs MSI \n{habit} \n{psd} \n{len(ds_arts.nray)} samples",
    fontsize=16,
    y=1.2,
    x=0.45,
)
# %% different PSD in the same plot
# loading data from scratch !!
# ch = 3
bins = np.linspace(90, 310, 100)
fig, ax = plt.subplots(1, 4, figsize=(15, 5), sharex=True, sharey=True)
for ch in range(4):
    hist_data = []
    for i in range(3):
        habit, psd, orbits, ds_arts = load_arts_output_data(
            i,
            2,
            file_pattern=file_pattern,
            n_files=80,  # Limit files for faster processing
        )
        hist, _ = np.histogram(ds_arts.isel(channel=ch).arts, bins=bins, density=True)
        hist_data.append(hist)
        ax[ch].semilogy(bins[:-1], hist_data[i], label=f"{habit}")
        ax[ch].set_xlabel(f"{ds_arts.channel[ch].values} Temperature [K]")
ax[0].set_ylabel("Probability Density")
ax[0].legend()
fig.suptitle(f"PDFs of simulated AWS \n{psd} \n{len(ds_arts.nray)} samples")

# %%
i = 1
mask = ds_arts.sel(channel=f"AWS4{i}").to_array().isnull().any("variable").compute()
ds_channel = ds_arts.sel(channel=f"AWS4{i}").where(~mask, drop=True)

# Get the data
x_data = ds_channel.arts.values
y_data = ds_channel.pixel_values.values

# Remove NaN values
valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
x_data = x_data[valid_mask]
y_data = y_data[valid_mask]

# Create 2D histogram data first
hist2d, xedges, yedges = np.histogram2d(x_data, y_data, bins=(bins_aws, bins_msi), density=True)

X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

# Compute percentiles for each histogram
h = hist2d.T
# levels = [np.percentile(h[h > 0], p) for p in percentiles]
# %%
# Create line contour plot with log normalization
cs = plt.contour(
    X,
    Y,
    np.where(h > 0, h, np.nan),
    # levels=np.logspace(-6, -2),
    levels=np.linspace(1e-6, 1e-2),
    # cmap="Reds",
)
cbar = plt.colorbar(
    cs,
    shrink=0.8,
    aspect=20,
)
