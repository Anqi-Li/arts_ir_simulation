# %%
import xarray as xr
from combine_CPR_MSI import (
    get_cpr_msi_from_orbits,
    read_xmet,
    xr_vectorized_height_interpolation,
    align_xmet_horizontal_grid,
)
import numpy as np
from tqdm import tqdm
from pathlib import Path
from physics.unitconv import mixing_ratio2mass_conc, dry_air_density


# %%
def prepare_arts(
    orbit_frame,
    remove_clearsky=True,
    lowest_dBZ_threshold=-30,
    vars_xmet=[
        "temperature",
        "pressure",
        "specific_humidity",
        "ozone_mass_mixing_ratio",
        "specific_cloud_liquid_water_content",
        "specific_rain_water_content",
    ],
    height_grid=np.arange(0, 20e3, 100),
):

    # load CPR and MSI data
    xds = get_cpr_msi_from_orbits(
        orbit_numbers=orbit_frame,
        msi_band=5,
        get_xmet=False,
        add_dBZ=True,
        remove_underground=True,
    )
    if remove_clearsky:
        # mask clearsky profiles
        mask_clearsky = (xds["dBZ"] < lowest_dBZ_threshold).all(dim="nbin")
        xds = xds.where(~mask_clearsky.compute(), drop=True)

    # % load xmet data
    ds_xmet = align_xmet_horizontal_grid(
        read_xmet(orbit_number=orbit_frame),
        xds,
    )

    # % Interpolate the data to a uniform height grid
    da_dBZ_height = xr_vectorized_height_interpolation(
        ds=xds,
        height_name="binHeight",
        variable_name="dBZ",
        height_grid=height_grid,
        height_dim="nbin",
        new_height_dim="height_grid",
    )

    ds_xmet["log_pressure"] = ds_xmet["pressure"].pipe(np.log)
    _vars_xmet = [v for v in vars_xmet if v != "pressure"] + ["log_pressure"]
    ds_xmet_height = xr_vectorized_height_interpolation(
        ds=ds_xmet,
        height_name="geometrical_height",
        variable_name=_vars_xmet,
        height_grid=height_grid,
        height_dim="height",
        new_height_dim="height_grid",
    )
    ds_xmet_height[["log_pressure", "temperature"]] = ds_xmet_height[
        ["log_pressure", "temperature"]
    ].interpolate_na(
        dim="height_grid",
        method="linear",
        fill_value="extrapolate",
    )
    ds_xmet_height["pressure"] = ds_xmet_height["log_pressure"].pipe(np.exp)

    for v in vars_xmet:
        ds_xmet_height[v] = ds_xmet_height[v].assign_attrs(ds_xmet[v].attrs)

    # make liquid water content
    # assuming dry air, ideal gas law, and constant specific gas constant for air
    # R = 287 J/(kg K) = 287e3 m^2
    # ds_xmet_height = ds_xmet_height.eval("density = pressure/(temperature * 287e3)")
    ds_xmet_height["density"] = dry_air_density(
        ds_xmet_height["pressure"],
        ds_xmet_height["temperature"],
    )
    ds_xmet_height["density"] = ds_xmet_height["density"].assign_attrs(
        dict(
            long_name="Air Density",
            units="kg m-3",
        )
    )
    # ds_xmet_height = ds_xmet_height.eval(
    #     "cloud_liquid_water_content = specific_cloud_liquid_water_content * density"
    # )
    ds_xmet_height["cloud_liquid_water_content"] = mixing_ratio2mass_conc(
        ds_xmet_height["specific_cloud_liquid_water_content"],  # kg/kg
        ds_xmet_height["pressure"],  # Pa
        ds_xmet_height["temperature"],  # K
    ).assign_attrs(
        dict(
            long_name="Cloud Liquid Water Content",
            units="kg m-3",
        )
    )

    ds_xmet_height["cloud_liquid_water_content"] = (
        ds_xmet_height["cloud_liquid_water_content"]
        # .assign_attrs(
        #     dict(
        #         long_name="Cloud Liquid Water Content",
        #         units="kg m-3",
        #     )
        # )
        .where(
            ds_xmet_height["cloud_liquid_water_content"] >= 0,
            0,  # ensure no negative values
        )
    )

    # ds_xmet_height = ds_xmet_height.eval(
    #     "rain_water_content = specific_rain_water_content * density"
    # )
    ds_xmet_height["rain_water_content"] = mixing_ratio2mass_conc(
        ds_xmet_height["specific_rain_water_content"],  # kg/kg
        ds_xmet_height["pressure"],  # Pa
        ds_xmet_height["temperature"],  # K
    ).assign_attrs(
        dict(
            long_name="Rain Liquid Water Content",
            units="kg m-3",
        )  # kg/m^3 = kg/kg * kg/m^3
    )
    ds_xmet_height["rain_water_content"] = (
        ds_xmet_height["rain_water_content"]
        # .assign_attrs(
        #     dict(
        #         long_name="Rain Liquid Water Content",
        #         units="kg m-3",
        #     )
        # )
        .where(
            ds_xmet_height["rain_water_content"] >= 0,
            0,  # ensure no negative values
        )
    )
    vars_xmet = [
        v.replace(
            "specific_cloud_liquid_water_content", "cloud_liquid_water_content"
        ).replace("specific_rain_water_content", "rain_water_content")
        for v in vars_xmet
    ]

    # % filter bad pixels, cleary sky, etc.
    # mask extreme MSI values
    mask_bad_msi = np.logical_or(
        (xds["pixel_values"] > 500), (xds["pixel_values"] < 150)
    )

    mask = ~mask_bad_msi

    # % combine all necessary variables
    ds = (
        xds[["surfaceElevation", "pixel_values"]]
        .assign(
            dBZ=da_dBZ_height,
            **{
                v: (da_dBZ_height.dims, ds_xmet_height[v].data, ds_xmet_height[v].attrs)
                for v in vars_xmet
            },
        )
        .where(mask.compute(), drop=True)
    )

    return ds


# %%
if __name__ == "__main__":
    # %%
    # orbit_frame = "01162E"
    for orbit_frame in tqdm(
        [
            "01162E",
            "04090A",
            "04758A",
            "03994G",
            "04142B",
            "04937G",
            "05248E",
            "04688D",
        ],
        desc="Processing orbits",
    ):

        path_save = "../data/earthcare/arts_input_data2/"
        filename_save = f"arts_input_{orbit_frame}.nc"
        filepath = Path(path_save) / filename_save
        if filepath.exists():
            print(f"File {filename_save} already exists, skipping.")
            continue
        ds = prepare_arts(orbit_frame, remove_clearsky=False)
        ds.reset_index("nray").set_xindex("time").to_netcdf(
            path_save + filename_save,
            compute=True,
        )

# %%
from matplotlib import pyplot as plt

orbits = [
    "01162E",
    "04090A",
    "04758A",
    "03994G",
    "04142B",
    "04937G",
    "05248E",
    "04688D",
]
path = "../data/earthcare/arts_input_data2/"
ds = xr.open_mfdataset(
    path + "arts_input_*.nc",
    combine="nested",
    concat_dim="nray",
)
ds.cloud_liquid_water_content.pipe(np.log10).assign_attrs(
    units="Log10(kgm-3)"
).plot.hist(
    bins=100,
    range=(-10, 0),
    density=True,
)
# ds.rain_water_content.pipe(np.log10).plot.hist(
#     bins=100,
#     range=(-10, 0),
#     density=True,
# )
plt.title(
    f"""
          Log10 of Cloud Liquid Water Content
from orbits: {orbits}
"""
)
