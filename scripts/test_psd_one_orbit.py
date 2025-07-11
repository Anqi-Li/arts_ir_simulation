# %%
import numpy as np

from onion_table import *
from ir_earthcare import *
from plotting import plot_fwc_and_temperatures, plot_psd_mgd

# %%
habit = habit_std_list[0]
psd = psd_list[-1]  # "exponential" is the last one in the list
print(f"Selected habit: {habit}, PSD: {psd}")
coef_mgd = (
    {
        "n0": 1e10,  # Number concentration
        "ga": 1.5,  # Gamma parameter
        "mu": 0,  # Default value for mu
    }
    if psd == "Exponential"
    else None
)

ds_onion_invtable = get_ds_onion_invtable(
    habit=habit,
    psd=psd,
    coef_mgd=coef_mgd,
)
plot_table(ds_onion_invtable)
if psd == "Exponential":
    plot_psd_mgd(coefs_mgd=coef_mgd)

#%%
from pyarts.workspace import Workspace
x_scale = "linear"
y_scale = "log"
ws = Workspace(verbosity=0)
ws.psd_size_grid = np.linspace(1e-6, 50e-6)
ws.dpnd_data_dx_names = []

# Setting all 4 parameters by GIN
# This generates a single PSD
ws.pnd_agenda_input_t = np.array([190])  # Not really used
ws.pnd_agenda_input = np.array([[]])
ws.pnd_agenda_input_names = []
ws.psdDelanoeEtAl14(iwc=1, n0star=-999, Dm=-999)
plt.figure()
plt.plot(
    ws.psd_size_grid.value * 1e6,
    ws.psd_data.value[0],
    label=psd,
    c="C0",
)
plt.yscale(y_scale)
plt.xscale(x_scale)
plt.xlabel("Diameter (µm)")
plt.ylabel("PSD (µm⁻¹ cm⁻³)")
plt.legend()
plt.grid()
plt.show()
# %% take an earthcare input dataset
orbit_frame = "03872A"
path_earthcare = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data/earthcare/arts_input_data/",
)
ds_earthcare_ = xr.open_dataset(path_earthcare + f"arts_input_{orbit_frame}.nc")
ds_earthcare_ = get_frozen_water_content(ds_onion_invtable, ds_earthcare_)
ds_earthcare_ = get_frozen_water_path(ds_earthcare_)

ds_earthcare_["reflectivity_integral_log10"] = (
    ds_earthcare_["dBZ"]
    .pipe(lambda x: 10 ** (x / 10))
    .fillna(0)
    .integrate("height_grid")
    .pipe(np.log10)
)
mask = ds_earthcare_["reflectivity_integral_log10"] > 3.5
ds_earthcare_subset = ds_earthcare_.where(mask, drop=True).isel(
    nray=slice(None, None, 5),  # skip every xth ray to reduce computation time
)
print(f"Number of nrays: {len(ds_earthcare_subset.nray)}")

# % plot data
fig, ax = plot_fwc_and_temperatures(ds_earthcare_subset, fwc_threshold=1e-5)
# add some text for the chosen habit, psd and coef_mgd below the plots
ax[1].text(
    0.05,
    -0.3,
    f"Habit: {habit}, PSD: {psd}"
    + (
        f"n0: {coef_mgd['n0']:.1e}, ga: {coef_mgd['ga']}"
        if psd == "exponential"
        else ""
    ),
    transform=ax[1].transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.5),
)

# add text annotation for fwc_threshold
ax[1].text(
    0.05,
    -0.5,
    f"FWC Threshold for cloud definition: {ds_earthcare_subset['cloud_top_height'].attrs['fwc_threshold']:.1e} kg/m³",
    transform=ax[1].transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.5),
)
plt.show()


# %% process
def process_nray(i):
    return cal_y_arts(
        ds_earthcare_subset.isel(nray=i),
        habit,
        psd,
        coef_mgd=coef_mgd,
    )


y = []
bulkprop = []
vmr = []
auxiliary = []

with ProcessPoolExecutor(max_workers=64) as executor:
    futures = [
        executor.submit(process_nray, i) for i in range(len(ds_earthcare_subset.nray))
    ]
    for f in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="process nrays",
        file=sys.stdout,
        dynamic_ncols=True,
    ):
        da_y, da_bulkprop, da_vmr, da_auxiliary = f.result()
        y.append(da_y)
        bulkprop.append(da_bulkprop)
        vmr.append(da_vmr)
        auxiliary.append(da_auxiliary)

da_y = xr.concat(y, dim="nray")
da_bulkprop = xr.concat(bulkprop, dim="nray")
da_vmr = xr.concat(vmr, dim="nray")
da_auxiliary = xr.concat(auxiliary, dim="nray")

# %%
ds_arts = da_auxiliary.assign(
    arts=da_y,
    bulkprop=da_bulkprop,
    vmr=da_vmr,
)  # .assign(ds_earthcare_subset)
if len(ds_arts.nray) > 1:
    ds_arts = ds_arts.sortby("time")

ds_arts = xr.merge([ds_arts, ds_earthcare_subset])

# %%
ds_arts["arts_y_mean"] = ds_arts["arts"].mean("f_grid")
fig, ax = plot_fwc_and_temperatures(
    ds_arts,
    fwc_threshold=4e-5,
    temperature_vars=["pixel_values", "cloud_top_T", "arts_y_mean"],
)

# %%
