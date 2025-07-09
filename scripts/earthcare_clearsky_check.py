# %%
from cProfile import label
from turtle import title

from requests import get
from src.ir_earthcare import *

# %%
i, j = 0, 0
orbit_frame = "03994G"
habit_std = habit_std_list[i]  # Habit to use
psd = psd_list[j]  # PSD to use
print(habit_std)
print(psd)
# %% Load Earthcare data
ds_onion_invtable = xr.open_dataset(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        f"data/onion_invtables/onion_invtable_{habit_std}_{psd}.nc",
    ),
)[f"onion_invtable_{habit_std}_{psd}"]

# take an earthcare input dataset
path_earthcare = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data/earthcare/arts_input_data/",
)
ds_earthcare_ = xr.open_dataset(path_earthcare + f"arts_input_{orbit_frame}.nc")

# put FWC
ds_earthcare_ = get_frozen_water_content(ds_onion_invtable, ds_earthcare_)

# mask clearsky profiles
mask_no_fwc = (ds_earthcare_["frozen_water_content"] == 0).all(dim="height_grid")
ds_earthcare_clearsky = ds_earthcare_.where(mask_no_fwc, drop=True)
print(f"Number of nrays (clearsky): {len(ds_earthcare_clearsky.nray)}")


# %%
# mask high and cold cloud profiles
def get_cloud_top_height(ds, fwc_threshold=1e-5):
    """Calculate the cloud top height based on the frozen water content."""
    return (
        ds["height_grid"]
        .where(ds["frozen_water_content"] > fwc_threshold)
        .max("height_grid")
    )


ds_earthcare_["cloud_top_height"] = get_cloud_top_height(ds_earthcare_)
mask_high_cloud = (ds_earthcare_["cloud_top_height"] > 5e3) & (
    ds_earthcare_["pixel_values"] < 245
)
ds_earthcare_highcloud = ds_earthcare_.where(mask_high_cloud, drop=True)
print(f"Number of nrays (high cloud): {len(ds_earthcare_highcloud.nray)}")

# %% Plot the whole orbit
fig, ax = plt.subplots(2, 1, constrained_layout=True, sharex=True)
kwargs = dict(x="nray")
ds_earthcare_["frozen_water_content"].pipe(np.log10).plot(ax=ax[0], **kwargs, vmin=-5)
ds_earthcare_["cloud_liquid_water_content"].pipe(np.log10).plot(
    ax=ax[1], **kwargs, vmin=-5
)

# %% loop over nrays
ds_earthcare_subset = ds_earthcare_clearsky.isel(nray=slice(None, None, 5))
print(f"Number of nrays: {len(ds_earthcare_subset.nray)}")


# %%
def process_nray(i):
    return cal_y_arts(
        ds_earthcare_subset.isel(nray=i),
        habit_std,
        psd,
    )


y = []
bulkprop = []
vmr = []
auxiliary = []

with ProcessPoolExecutor(max_workers=20) as executor:
    futures = [
        executor.submit(process_nray, i) for i in range(len(ds_earthcare_subset.nray))
    ]
    failed_indices = []
    for idx, f in enumerate(
        tqdm(as_completed(futures), total=len(futures), desc="process nrays")
    ):
        try:
            da_y, da_bulkprop, da_vmr, da_auxiliary = f.result()
            y.append(da_y)
            bulkprop.append(da_bulkprop)
            vmr.append(da_vmr)
            auxiliary.append(da_auxiliary)
        except Exception as e:
            print(f"Error processing nray index {idx}: {e}")
            failed_indices.append(idx)

print("Failed indices:", failed_indices)

da_y = xr.concat(y, dim="nray")
da_bulkprop = xr.concat(bulkprop, dim="nray")
da_vmr = xr.concat(vmr, dim="nray")
da_auxiliary = xr.concat(auxiliary, dim="nray")

ds_arts = (
    da_auxiliary.assign(
        arts=da_y,
        bulkprop=da_bulkprop,
        vmr=da_vmr,
    )
    .sortby("time")
    .assign(ds_earthcare_subset.drop_sel(nray=failed_indices))
).rename(height_grid="z")

# %% some filtering?
ds_arts["cloud_top_height"] = get_cloud_top_height(
    ds_arts.rename(z="height_grid"), fwc_threshold=1e-5
)
mask_high_cloud = (ds_arts["cloud_top_height"] > 5e3) & (ds_arts["pixel_values"] < 245)
ds_arts = ds_arts.where(mask_high_cloud, drop=True)

# %%
title = "clearsky sanity check"
fig, ax = plt.subplots(6, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
kwargs = dict(x="nray")

ds_arts.surfaceElevation.plot(ax=ax[0], color="k", lw=0.5, label="Surface Elevation")
im0 = ds_arts["dBZ"].plot.imshow(ax=ax[0], **kwargs)
ax[0].legend()

im1 = (
    ds_arts["bulkprop"]
    .sel(scat_species="LWC")
    .pipe(np.log10)
    .plot.imshow(ax=ax[1], vmin=-4, vmax=-1, **kwargs)
)

ds_arts["rain_water_content"].pipe(np.log10).plot.imshow(ax=ax[2], vmin=-4, **kwargs)

im3 = ds_arts["vmr"].sel(abs_species="H2O").plot.imshow(ax=ax[3], **kwargs)

im2 = ds_arts["temperature"].plot.imshow(ax=ax[-2], **kwargs)

ds_arts["pixel_values"].plot(ax=ax[-1], label="MSI")
ds_arts["arts"].mean("f_grid").plot(ax=ax[-1], label="ARTS")
# (ds_arts["arts"].mean("f_grid") - ds_arts["pixel_values"]).plot(ax=ax[-1].twinx(), label="ARTS - MSI", color="k", lw=0.5)
ax[-1].legend()

# Turn off xlabels for all but the last axis
for a in ax[:-1]:
    a.set_xlabel("")

fig.suptitle(
    f"""
    {title}
    {habit_std}
    {psd}
    {orbit_frame}
    """,
    fontsize=16,
)
# %%
