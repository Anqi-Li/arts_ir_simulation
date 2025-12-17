# %%
import numpy as np
from onion_table import *
from earthcare_ir import *
from ectools import ecio
from ectools import ecplot
import data_paths
from scipy.interpolate import NearestNDInterpolator

# %% take an earthcare input dataset
orbit_frame = "06356H"

run_arts = True
if run_arts:
    ds_arts_list = []
    for p in [PSD.D14, PSD.F07T, PSD.MDG]:
        for h in [Habit.Bullet]:
            print(f"Running ARTS simulation for PSD: {p}, Habit: {h}...")
            ds_arts = main(
                orbit_frame=orbit_frame,
                habit_list=[h],
                psd_list=[p],
                skip_profiles=1500,
                max_workers=32,
                save_results=False,
                skip_existing=False,
            ).assign_coords({"psd": p, "habit": h})
            ds_arts_list.append(ds_arts)

    ds_arts = xr.concat(ds_arts_list, dim="new_dim").set_xindex(["psd", "habit"]).unstack("new_dim")    
    ds_arts.encoding = {"source": "N/A"}

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
product_baseline = src_cpr.split("_")[1][-2:]
observation_datetime = src_cpr.split("_")[5]
production_datetime = src_cpr.split("_")[6]

ds_cfmr = ecio.load_CFMR(
    srcpath=data_paths.CFMR,
    product_baseline=product_baseline,
    observation_datetime=observation_datetime,
    production_datetime=production_datetime,
    frame=orbit_frame[-1],
    orbit=orbit_frame[:-1],
    nested_directory_structure=True,
)
ds_cfmr.close()
print("C-FMR loading done.")

if run_arts:
    ds_xmet = ecio.load_XMET(
        srcpath=data_paths.XMET,
        frame=orbit_frame[-1],
        orbit=orbit_frame[:-1],
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
    product_baseline="BA",
    frame=orbit_frame[-1],
    orbit=orbit_frame[:-1],
    nested_directory_structure=True,
)
ds_msi.close()
print("MSI loading done.")

# pick the nearest MSI pixel for each CPR ray
ds_msi_TIR2_select = ecio.get_MSI_from_footprint(
    ds_msi["TIR2"],
    ds_arts,
    combine_datasets=False,
)
print("Selecting nearest MSI TIR2 pixel for each CPR ray done.")


# %% plot comparison with CFMR reflectivity and MSI TIR2
ds_compare = ds_arts.assign({"msi": ("along_track", ds_msi_TIR2_select.data)})
ds_compare.encoding = ds_msi.encoding  # keep the original encoding info
# ds_compare = ds_compare.reindex_like(ds_cfmr).assign_coords(ds_cfmr.coords)
ds_compare["diff"] = ds_compare["arts"] - ds_compare["msi"]

nrows = 4
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
# habit = Habit.Bullet

for i, habit in enumerate(ds_compare.habit.data):
    ecplot.plot_EC_1D(
        axes[i + 1],
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
        legend_markerscale=2,
        use_localtime=False,
    )
    axes[i + 1].grid()
# axes[1].legend().remove()
# %%
