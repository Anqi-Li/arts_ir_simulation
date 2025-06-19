# %%
import xarray as xr
from combine_CPR_MSI import (
    get_cpr_msi_from_orbits,
    read_xmet,
    xr_vectorized_height_interpolation,
    align_xmet_horizontal_grid,
    read_cpr,
)
import numpy as np


# %%
def prepare_arts(
    orbit_frame,
    remove_clearsky=True,
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

    # load XMET data and align with CPR
    # ds_xmet = read_xmet(
    #     orbit_number=orbit_frame,
    #     base_path="../data/earthcare/AUX_MET_1D_aligned_CPR",
    #     group=None,
    # )
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
    )

    ds_xmet["log_pressure"] = ds_xmet["pressure"].pipe(np.log)
    _vars_xmet = [v for v in vars_xmet if v != "pressure"] + ["log_pressure"]
    ds_xmet_height = xr_vectorized_height_interpolation(
        ds=ds_xmet,
        height_name="geometrical_height",
        variable_name=_vars_xmet,
        height_grid=height_grid,
        height_dim="height",
    )
    ds_xmet_height["pressure"] = ds_xmet_height["log_pressure"].pipe(np.exp)

    # % filter bad pixels, cleary sky, etc.
    # mask clearsky profiles
    lowest_dBZ_threshold = -25
    mask_clearsky = (da_dBZ_height < lowest_dBZ_threshold).all(dim="height_grid")

    # mask extreme MSI values
    mask_bad_msi = np.logical_or(
        (xds["pixel_values"] > 500), (xds["pixel_values"] < 150)
    )

    # combine all masks
    if remove_clearsky:
        mask = np.logical_and(~mask_clearsky, ~mask_bad_msi)
    else:
        mask = ~mask_bad_msi

    # % combine all necessary variables
    ds = (
        xds[["surfaceElevation", "pixel_values"]]
        .assign(
            dBZ=da_dBZ_height,
            **{
                v: (da_dBZ_height.dims, ds_xmet_height[v].__array__())
                for v in vars_xmet
            },
        )
        .where(mask.compute(), drop=True)
    )
    for v in vars_xmet:
        ds[v].attrs = ds_xmet[v].attrs
    return ds


# %%
if __name__ == "__main__":
    # %%
    orbit_frame = "01162E"
    ds = prepare_arts(orbit_frame, remove_clearsky=False)

    # %% save ds to netcdf file
    path_save = "../data/earthcare/arts_x_data/"
    ds.reset_index("nray").set_xindex("time").to_netcdf(
        path_save + f"arts_x_{orbit_frame}.nc"
    )
