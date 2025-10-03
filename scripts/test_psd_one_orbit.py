# %%
import numpy as np
from onion_table import *
from earthcare_ir import *
from ectools import ecio
from ectools import ecplot
import data_paths

# %% take an earthcare input dataset
orbit_frame = "06554E"
ds_cfmr = ecio.get_XMET(
    XMET=ecio.load_XMET(
        srcpath=data_paths.XMET,
        frame_code=orbit_frame,
        nested_directory_structure=True,
    ),
    ds=ecio.load_CFMR(
        srcpath=data_paths.CFMR,
        prodmod_code="ECA_EXBA",
        frame_code=orbit_frame,
        nested_directory_structure=True,
    ),
    XMET_1D_variables=[],
    XMET_2D_variables=[
        "temperature",
        "pressure",
        "specific_humidity",
        "ozone_mass_mixing_ratio",
        "specific_cloud_liquid_water_content",
    ],
).set_coords(["time", "latitude", "longitude", "height", "surface_elevation"])

print("C-FMR and X-MET loading done.")

# %% Remove profiles with all NaN or very low reflectivity
nan_pressure = ds_cfmr["pressure"].isnull().all(dim="CPR_height")
nan_height = ds_cfmr["height"].isnull().all(dim="CPR_height")
low_reflectivity = (ds_cfmr["reflectivity_corrected"].fillna(-999) < -15).all(dim="CPR_height")
mask = ~(nan_pressure | nan_height | low_reflectivity)

ds_cfmr_subset = ds_cfmr.where(mask, drop=True).isel(
    along_track=slice(None, None, 10),  # skip every xth ray to reduce computation time
)
ds_cfmr_subset.encoding = ds_cfmr.encoding  # keep the original encoding info
print(f"Subset number of profiles: {len(ds_cfmr_subset.along_track)}")

# reverse the height dimension so that pressure is increasing
ds_cfmr_subset = ds_cfmr_subset.isel(CPR_height=slice(None, None, -1))

# %% Prepare h2o_volume_mixing_ratio and liquid_water_content
ds_cfmr_subset = get_lwc_and_h2o_vmr(ds_cfmr_subset)
print("LWC and H2O VMR preparation done.")

# %% Invertion to get frozen water content
habit = habit_std_list[0]
psd = psd_list[0]
print(f"Selected habit: {habit}, PSD: {psd}")
coef_mgd = (
    {
        "n0": 1e10,  # Number concentration
        "ga": 1.5,  # Gamma parameter
        "mu": 0,  # Default value for mu
    }
    if psd == "ModifiedGamma"
    else None
)

ds_cfmr_subset = get_frozen_water_content(
    get_ds_onion_invtable(habit=habit, psd=psd, coef_mgd=coef_mgd),
    ds_cfmr_subset,
)
print("FWC inversion done.")


# %% ARTS simulation for each profile
def process_nray(i):
    return cal_y_arts(
        ds_cfmr_subset.isel(along_track=i),
        habit,
        psd,
        coef_mgd=coef_mgd,
    )


y = []
auxiliary = []

with ProcessPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(process_nray, i) for i in range(len(ds_cfmr_subset.along_track))]
    for f in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="process profiles",
        file=sys.stdout,
        dynamic_ncols=True,
    ):
        da_y, _, _, da_auxiliary = f.result()
        y.append(da_y)
        auxiliary.append(da_auxiliary)

print("ARTS simulation done.")

# %% concatenate results
da_y = xr.concat(y, dim="along_track")
da_auxiliary = xr.concat(auxiliary, dim="along_track")

ds_arts = da_auxiliary.assign({"arts": da_y.mean("f_grid")})
ds_arts["arts"].attrs.update(
    {
        "long_name": "ARTS simulated brightness temperature",
        "units": "K",
        "CPR source": ds_cfmr.encoding["source"].split("/")[-1],
        "habit": habit,
        "PSD": psd,
        "coef_mgd": str(coef_mgd) if coef_mgd is not None else "None",
    }
)
if len(ds_arts.along_track) > 1:
    ds_arts = ds_arts.sortby("along_track")
print("Concatenating results done.")

# %%
ds_cfmr = ds_cfmr.merge(ds_arts)
print("Merging ARTs result with input dataset (C-FMR) done.")

# %% Compare with MSI TIR2
# load MSI TIR2 data
ds_msi = ecio.load_MRGR(
    srcpath="/data/s6/L1/EarthCare/L1/MSI_RGR_1C",
    prodmod_code="ECA_EXBA",
    frame_code=orbit_frame,
    nested_directory_structure=True,
)

# %% pick the nearest MSI pixel for each CPR ray
from scipy.interpolate import NearestNDInterpolator

flatten_hcoords_msi = (
    ds_msi.reset_coords(["longitude", "latitude"])[["longitude", "latitude"]]
    .stack({"horizontal_grid": ["along_track", "across_track"]})
    .to_array()
)
NearestIndex = NearestNDInterpolator(flatten_hcoords_msi.data.T, np.arange(len(flatten_hcoords_msi["horizontal_grid"])))
nearest_indices_on_flatten_hcoords_msi = NearestIndex(np.array([ds_arts["longitude"], ds_arts["latitude"]]).T).astype(int)
ds_msi_TIR2_select = (
    ds_msi["TIR2"]
    .stack({"horizontal_grid": ["along_track", "across_track"]})
    .isel({"horizontal_grid": nearest_indices_on_flatten_hcoords_msi})
)

ds_compare = ds_arts.assign({"msi": ("along_track", ds_msi_TIR2_select.data)})
ds_compare.encoding = ds_msi.encoding  # keep the original encoding info

# %% plot comparison with CFMR reflectivity
nrows = 2
fig, axes = plt.subplots(figsize=(25, 7 * nrows), nrows=nrows, gridspec_kw={"hspace": 0.67}, sharex=True)
ecplot.plot_EC_2D(
    axes[0],
    ds_cfmr,
    "reflectivity_corrected",
    "Z",
    units="dBZ",
    plot_scale="linear",
    plot_range=[-35, 35],
    hmax=20e3,
    use_localtime=False,
    cmap="calipso",
)
ecplot.add_temperature(axes[0], ds_cfmr)
ecplot.add_marble(axes[0], ds_cfmr)

ds_compare = ds_compare.reindex_like(ds_cfmr).assign_coords(ds_cfmr.coords)
# ecplot.plot_EC_1D(
#     axes[1],
#     ds_compare,
#     {
#         "ARTS": {"xdata": ds_compare["time"], "ydata": ds_compare["arts"], "marker": "o", "color": "r"},
#         "MSI TIR2": {"xdata": ds_compare["time"], "ydata": ds_compare["msi"], "marker": "o", "color": "k"},
#     },
#     "IR temperature",
#     r"$T_B$ [K]",
#     timevar="time",
#     include_ruler=False,
# )

ecplot.plot_EC_1D(
    axes[1],
    ds_compare,
    {
        "diff(ARTS - MSI)": {
            "xdata": ds_compare["time"],
            "ydata": ds_compare["arts"] - ds_compare["msi"],
            "marker": "*",
            "markersize": 8,
            "color": "b",
        },
    },
    "diff(ARTS - MSI)",
    r"$T_B$ [K]",
    timevar="time",
    include_ruler=False,
)
axes[1].grid()
axes[1].legend().remove()
# %%
