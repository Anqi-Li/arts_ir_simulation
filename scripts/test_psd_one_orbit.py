# %%
import numpy as np
from onion_table import *
from earthcare_ir import *
from ectools import ecio
from ectools import ecplot
import data_paths
from scipy.interpolate import NearestNDInterpolator

# %% take an earthcare input dataset
orbit_frame = "06554E"

run_arts = False
if run_arts:
    # run ARTS simulation for one orbit
    ds_arts = main(
        orbit_frame=orbit_frame,
        habit_list=[Habit.Bullet],
        psd_list=[PSD.MDG],
        skip_profiles=500,
        max_workers=32,
        save_results=False,
        skip_existing=False,
    )
    print("ARTS simulation done.")
else:
    ds_arts = (
        xr.open_mfdataset(
            os.path.join(data_paths.arts_output_TIR2, f"*{orbit_frame}*.nc"),
            concat_dim="new_dim",
            combine="nested",
            parallel=True,
            preprocess=lambda ds: ds.assign_coords(
                {"psd": ds.arts.attrs["PSD"], "habit": ds.arts.attrs["habit"]}
            ),
        )
        .set_xindex(["psd", "habit"])
        .unstack("new_dim")
    )
    print("Loading existing ARTS result done.")
    ds_arts.close()

# %%
src_cpr = ds_arts.arts.attrs["CPR source"]
prodmod_code = src_cpr[:8]
product_code = src_cpr[9:19]
frame_datetime = src_cpr.split("_")[5]
production_datetime = src_cpr.split("_")[6]

ds_cfmr = ecio.load_CFMR(
    srcpath=data_paths.CFMR,
    prodmod_code=prodmod_code,
    product_code=product_code,
    frame_datetime=frame_datetime,
    production_datetime=production_datetime,
    frame_code=orbit_frame,
    nested_directory_structure=True,
)
ds_cfmr.close()
print("C-FMR loading done.")

if not run_arts:
    ds_xmet = ecio.load_XMET(
        srcpath=data_paths.XMET,
        frame_code=orbit_frame,
        nested_directory_structure=True,
    )
    ds_xmet.close()
    print("X-MET loading done.")
    # merge XMET data into ds_cfmr
    ds_cfmr = ecio.get_XMET(
        ds_xmet,
        ds_cfmr,
        XMET_1D_variables=[],
        XMET_2D_variables=["temperature"],
    )
    print("Merging XMET data into input dataset (C-FMR) done.")


ds_cfmr = ds_cfmr.set_coords(
    ["time", "latitude", "longitude", "height", "surface_elevation"]
).merge(ds_arts)
print("Merging ARTs result with input dataset (C-FMR) done.")

# %% Compare with MSI TIR2
# load MSI TIR2 data
ds_msi = ecio.load_MRGR(
    srcpath=data_paths.MRGR,
    prodmod_code="ECA_EXBA",
    frame_code=orbit_frame,
    nested_directory_structure=True,
)
ds_msi.close()
print("MSI loading done.")

# pick the nearest MSI pixel for each CPR ray
flatten_hcoords_msi = (
    ds_msi.reset_coords(["longitude", "latitude"])[["longitude", "latitude"]]
    .stack({"horizontal_grid": ["along_track", "across_track"]})
    .to_array()
)
NearestIndex = NearestNDInterpolator(
    flatten_hcoords_msi.data.T, np.arange(len(flatten_hcoords_msi["horizontal_grid"]))
)
nearest_indices_on_flatten_hcoords_msi = NearestIndex(
    np.array([ds_arts["longitude"], ds_arts["latitude"]]).T
).astype(int)
ds_msi_TIR2_select = (
    ds_msi["TIR2"]
    .stack({"horizontal_grid": ["along_track", "across_track"]})
    .isel({"horizontal_grid": nearest_indices_on_flatten_hcoords_msi})
)
print("Selecting nearest MSI TIR2 pixel for each CPR ray done.")


# %% plot comparison with CFMR reflectivity

ds_compare = ds_arts.assign({"msi": ("along_track", ds_msi_TIR2_select.data)})
ds_compare.encoding = ds_msi.encoding  # keep the original encoding info
# ds_compare = ds_compare.reindex_like(ds_cfmr).assign_coords(ds_cfmr.coords)
ds_compare["diff"] = ds_compare["arts"] - ds_compare["msi"]

nrows = 2
fig, axes = plt.subplots(
    figsize=(25, 7 * nrows), nrows=nrows, gridspec_kw={"hspace": 0.67}, sharex=True
)

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

# psd = PSD.MDG
habit = Habit.Bullet
ecplot.plot_EC_1D(
    axes[1],
    ds_compare,
    {
        f"{psd}": {
            "xdata": ds_compare["time"],
            "ydata": ds_compare["diff"].sel(habit=habit, psd=psd),
            "marker": ".",
            "markersize": 10,
        } 
        # for habit in ds_compare.habit.data
        for psd in ds_compare.psd.data
    },
    title=f"Diff(ARTS - MSI) for habit: {habit}",
    ylabel=r"$T_B$ [K]",
    timevar="time",
    legend_markerscale=2
)
axes[1].grid()
# axes[1].legend().remove()
# %%
