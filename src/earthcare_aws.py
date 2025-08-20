# %%
from sys import argv
import sys
from earthcare_ir import insert_atm_from_earthcare
import pyarts
import numpy as np
import os
import xarray as xr
from tqdm import tqdm
import easy_arts.wsv as wsv
import easy_arts.easy_arts as ea
from easy_arts.data_model import (
    CloudboxOption,
    SpectralRegion,
    AbsSpeciesPredefinedOption,
)
from earthcare_ir import insert_bulkprop_from_earthcare, get_frozen_water_content
from concurrent.futures import ProcessPoolExecutor, as_completed
from onion_table import *

habit_std_list = ["LargePlateAggregate", "8-ColumnAggregate", "6-BulletRosette"]
psd_list = ["DelanoeEtAl14", "FieldEtAl07TR", "Exponential"]


# %%
def cal_y_arts(ds_earthcare, habit_std, psd, coef_mgd=None):
    """
    Simulate ARTS brightness temperature for a given Earthcare profile, ice habit, and PSD.

    Parameters:
        ds_earthcare (xarray.Dataset): Input dataset containing atmospheric and cloud properties.
        habit_std (str): Ice crystal habit name.
        psd (str): Particle size distribution identifier.
        coef_mgd (dict, optional): Coefficients for exponential PSD (if required).

    Returns:
        tuple: (Brightness temperature, bulk property field, VMR field, auxiliary data) as xarray.DataArrays.
    """
    # Create a workspace
    ws = ea.wsInit(SpectralRegion.SUBMM)

    # Download ARTS catalogs if they are not already present.
    pyarts.cat.download.retrieve()
    ws.stdhabits_folder = "/scratch/patrick/Data/StandardHabits"
    ws.SetNumberOfThreads(nthreads=1)

    ws.iy_main_agendaSet(option="Emission")
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
    ws.surface_rtprop_agendaSet(option="Blackbody_SurfTFromt_field")
    ws.iy_space_agendaSet(option="CosmicBackground")

    ws.ppath_agendaSet(option="FollowSensorLosPath")
    ws.ppath_step_agendaSet(option="GeometricPath")
    ws.water_p_eq_agendaSet(option="MK05")
    ws.PlanetSet(option="Earth")

    # Our output unit
    ws.iy_unit = "PlanckBT"  # Planck brightness temperature

    ws.stokes_dim = 1
    ws.atmosphere_dim = 1
    ws.sensor_pos = [[300e3]]
    ws.sensor_los = [[170]]
    ws.p_grid = ds_earthcare["pressure"].data  # Pa
    ws.lat_grid = []
    ws.lon_grid = []
    ws.f_grid = aws_f_grid().data

    # Set absorption
    ea.abs_speciesPredefined(
        ws, add_lwc=True, option=AbsSpeciesPredefinedOption.RTTOV_v13x
    )

    ws.propmat_clearsky_agendaAuto()
    ws.jacobianOff()
    ws.sensorOff()

    # Set VMR field
    insert_atm_from_earthcare(ds_earthcare, ws)

    # % Set the cloud layer
    scat_species = ["FWC"]
    insert_bulkprop_from_earthcare(
        ds_earthcare,
        ws,
        habit_std,
        psd,
        scat_species,
        coef_mgd=coef_mgd,
        habit_pingyang=False,
    )

    # Cloudbox
    ea.scat_data(ws)
    ea.cloudbox(ws, CloudboxOption.AUTO)
    ws.pnd_fieldCalcFromParticleBulkProps()

    ea.checks(ws)
    ea.disort(ws, Npfct=-1)
    ws.yCalc()

    return extract_xr_arrays(ds_earthcare, ws, scat_species)


def aws_f_grid():
    f_center = 325.150e9
    f_aws41 = 1.2e9
    f_aws42 = 2.4e9
    f_aws43 = 4.1e9
    f_aws44 = 6.6e9
    f_grid = [
        f_center + f_aws41,
        f_center - f_aws41,
        f_center + f_aws42,
        f_center - f_aws42,
        f_center + f_aws43,
        f_center - f_aws43,
        f_center + f_aws44,
        f_center - f_aws44,
    ]
    channel_names = [
        "AWS41",
        "AWS41",
        "AWS42",
        "AWS42",
        "AWS43",
        "AWS43",
        "AWS44",
        "AWS44",
    ]
    f_grid_da = xr.DataArray(
        f_grid,
        dims=["channel"],
        coords={"channel": channel_names},
        name="f_grid",
        attrs={"units": "Hz", "description": "AWS channel frequencies"},
    )
    return f_grid_da.sortby(lambda x: x)


def extract_xr_arrays(ds_earthcare, ws, scat_species):
    da_y = (
        xr.DataArray(
            name="Brightness Temperature",
            data=ws.y.value,
            dims=["f_grid"],
            coords={
                "f_grid": ws.f_grid.value,
                "channel": ("f_grid", aws_f_grid().channel.data),
            },
            attrs={
                "unit": "K",
                "long_name": "Brightness Temperature",
            },
        )
        .groupby("channel")
        .mean()
    )

    da_bulkprop_field = xr.DataArray(
        name="particle_bulkprob_field",
        data=np.atleast_2d(ws.particle_bulkprop_field.value.value.squeeze()),
        coords={
            "scat_species": scat_species,
            "height_grid": ds_earthcare["height_grid"],
        },
        attrs={
            "unit": "kg m-3",
        },
    )
    da_vmr_field = xr.DataArray(
        name="vmr_field",
        data=ws.vmr_field.value.value.squeeze(),
        coords={
            "abs_species": [str(s).split("-")[0] for s in list(ws.abs_species.value)],
            "height_grid": ds_earthcare["height_grid"],
        },
        attrs={
            "unit": "1",
            "long_name": "absoption species vmr",
        },
    )
    da_auxiliary = ds_earthcare[["time", "latitude", "longitude"]].copy()

    return da_y, da_bulkprop_field, da_vmr_field, da_auxiliary


# %%
if __name__ == "__main__":

    # take system arguments to select habit, psd and orbit frame
    if len(argv) == 4:
        i = int(argv[1])
        j = int(argv[2])
        orbit_frame = argv[3]
    else:
        raise ValueError(
            "Please provide habit and psd indices as command line arguments."
        )

    # %% choose invtable
    habit_std = habit_std_list[i]  # Habit to use
    psd = psd_list[j]  # PSD to use
    print(habit_std)
    print(psd)

    file_save_ncdf = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        f"data/earthcare/arts_output_aws_data/aws_2nd_{habit_std}_{psd}_{orbit_frame}.nc",
    )

    # %% check if the file already exists
    if os.path.exists(file_save_ncdf):
        print(f"File {file_save_ncdf} already exists. Skipping computation.")
        sys.exit(0)

    # %% Load EarthCare data
    path_earthcare = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data/earthcare/arts_input_data/",
    )
    ds_earthcare_ = xr.open_dataset(path_earthcare + f"arts_input_{orbit_frame}.nc")

    # Calculate onion table
    if psd == "Exponential":
        # coefficients for the exponential PSD
        coef_mgd = {
            "n0": 1e10,  # Number concentration
            "ga": 1.5,  # Gamma parameter
        }
    else:
        coef_mgd = None
    ds_onion_invtable = get_ds_onion_invtable(
        habit=habit_std,
        psd=psd,
        coef_mgd=coef_mgd,
    )

    
    ds_earthcare_subset = ds_earthcare_.isel(
        nray=slice(None, None, 2),  # skip every xth ray to reduce computation time
    )
    print(f"Number of nrays: {len(ds_earthcare_subset.nray)}")

    # put FWC
    ds_earthcare_subset = get_frozen_water_content(ds_onion_invtable, ds_earthcare_subset)

    # %% loop over nrays
    def process_nray(i):
        return cal_y_arts(
            ds_earthcare_subset.isel(nray=i),
            habit_std,
            psd,
            coef_mgd=coef_mgd,
        )

    # %%
    y = []
    bulkprop = []
    vmr = []
    auxiliary = []

    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(process_nray, i)
            for i in range(len(ds_earthcare_subset.nray))
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

    ds_arts = xr.combine_by_coords(
        [
            ds_arts.set_xindex("time"),
            ds_earthcare_subset.set_xindex("time"),
        ],
        join="inner",
    )
    ds_arts = ds_arts.sortby("time")

    # %% save to file
    ds_arts.to_netcdf(file_save_ncdf)
