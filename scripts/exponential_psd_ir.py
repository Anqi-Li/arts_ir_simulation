# %%
import numpy as np
from cycler import cycler

from onion_table import *
from ir_earthcare import *

# %%
habit = "LargePlateAggregate"
psd = "Exponential"
coefs_exp = {
    "n0": 10**10,  # Number concentration
    "ga": 1.5,  # Gamma parameter
}
ds_onion_invtable = get_ds_table(
    habit=habit,
    psd=psd,
    ws=make_onion_invtable(
        habit=habit,
        psd=psd,
        coefs_exp=coefs_exp,
    ),
)
# plot_table(ds_onion_invtable)
# %% take an earthcare input dataset

orbit_frame = "03872A"
path_earthcare = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data/earthcare/arts_input_data/",
)
ds_earthcare_ = xr.open_dataset(path_earthcare + f"arts_input_{orbit_frame}.nc")
ds_earthcare_ = get_frozen_water_content(ds_onion_invtable, ds_earthcare_)
ds_earthcare_["frozen_water_path"] = (
    ds_earthcare_["frozen_water_content"]
    .integrate("height_grid")
    .assign_attrs(
        units="kg/m^2",
        long_name="Frozen Water Path",
        description="Integrated frozen water content over height grid",
    )
)
# fwc_threshold = 1e-5
# ds_earthcare_ = get_cloud_top_T(ds_earthcare_, fwc_threshold=fwc_threshold)
mask = (
    ds_earthcare_["frozen_water_path"]
    > 5
    # & (ds_earthcare_["cloud_top_T"] < 245)
    # & (ds_earthcare_["pixel_values"] < 245)
    # & (ds_earthcare_["frozen_water_content"] > 1e-5).any("height_grid")
)
ds_earthcare_subset = ds_earthcare_.where(mask, drop=True).isel(
    nray=slice(None, None, 5),  # skip every xth ray to reduce computation time
)
print(f"Number of nrays: {len(ds_earthcare_subset.nray)}")


# %%
def plot_frozen_water_and_temperature(
    ds_earthcare_subset,
    fwc_threshold=None,
    temperature_vars=["pixel_values", "cloud_top_T"],
    add_diff=False,
):
    if fwc_threshold is None and "cloud_top_T" not in ds_earthcare_subset:
        raise ValueError(
            "Please provide a fwc_threshold or ensure 'cloud_top_T' is in the dataset."
        )
    if fwc_threshold is not None:
        ds_earthcare_subset = get_cloud_top_T(
            ds_earthcare_subset, fwc_threshold=fwc_threshold
        )

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6), constrained_layout=True)
    ds_earthcare_subset["frozen_water_content"].pipe(np.log10).plot(
        ax=ax[0],
        x="nray",
        y="height_grid",
        cmap="viridis",
        cbar_kwargs={"label": "Frozen Water Content log10(kg/m^3)"},
    )
    ds_earthcare_subset["cloud_top_height"].plot(
        ax=ax[0],
        x="nray",
        label="Cloud Top Height",
        color="orange",
    )

    ds_earthcare_subset[temperature_vars].to_array().plot.line(
        ax=ax[1],
        x="nray",
        hue="variable",
        add_legend=True,
    )
    if add_diff:
        ax1_twinx = ax[1].twinx()
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        new_color_cycle = cycler(color=default_colors[1:])
        ax1_twinx.set_prop_cycle(new_color_cycle)

        ds_earthcare_subset[temperature_vars].to_array('diff').diff('diff').plot.line(
            ax=ax1_twinx,
            x="nray",
            hue="diff",
            add_legend=True,
            linestyle="--",
            alpha=0.5,
        )
        ax1_twinx.set_ylabel("Temperature Difference (K)") 

    
    ax[0].set_ylabel("Height (m)")
    ax[0].set_title("Frozen Water Content")
    ax[0].legend()
    ax[1].set_xlabel("Ray Index")
    ax[1].set_ylabel("Temperature (K)")
    ax[1].set_title("Temperatures")
    
    # add some text for the chosen habit, psd and coefs_exp below the plots
    ax[1].text(
        0.05,
        -0.3,
        f"Habit: {habit}, PSD: {psd}, n0: 1e{np.log10(coefs_exp['n0']):.1f}, ga: {coefs_exp['ga']}",
        transform=ax[1].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    
    # add text annotation for fwc_threshold
    ax[1].text(
        0.05,
        -0.5,
        "FWC Threshold for cloud definition: 1e{:.2f} kg/m^3 ".format(
            np.log10(ds_earthcare_subset["cloud_top_height"].attrs["fwc_threshold"])
        ),
        transform=ax[1].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.show()

plot_frozen_water_and_temperature(ds_earthcare_subset, fwc_threshold=1e-5)

# %%
# i = 50
# da_y, da_bulkprop, da_vmr, da_auxiliary = cal_y_arts(
#     ds_earthcare_subset.isel(nray=i),
#     habit,
#     psd,
#     coefs_exp=coefs_exp,
# )
# print("arts brightness temperature {} K".format(da_y.mean().data))
# print(f"pixel values {ds_earthcare_subset.pixel_values.isel(nray=i).data} K")
# print(f'n0: {coefs_exp["n0"]}, ga: {coefs_exp["ga"]}')

# %%
def process_nray(i):
    return cal_y_arts(
        ds_earthcare_subset.isel(nray=i),
        habit,
        psd,
        coefs_exp=coefs_exp,
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

ds_arts = da_auxiliary.assign(
    arts=da_y,
    bulkprop=da_bulkprop,
    vmr=da_vmr,
)
if len(ds_arts.nray) > 1:
    ds_arts = ds_arts.sortby("time")

ds_arts = xr.merge([ds_arts, ds_earthcare_subset])

# %%
ds_arts["arts_y_mean"] = ds_arts["arts"].mean("f_grid")
plot_frozen_water_and_temperature(
    ds_arts,
    fwc_threshold=10 ** (-4.5),
    temperature_vars=["pixel_values",  "arts_y_mean", "cloud_top_T"],
)

# %%
