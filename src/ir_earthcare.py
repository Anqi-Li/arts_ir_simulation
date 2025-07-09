# %%
from sys import argv
import sys
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
)
from physics.unitconv import specific_humidity2h2o_p
from physics.constants import DENSITY_H2O_LIQUID
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

# % map the standard habit to the Yang habit name
map_habit = {
    "LargePlateAggregate": "Plate-Smooth",
    "8-ColumnAggregate": "8-ColumnAggregate-Smooth",
}

habit_std_list = ["LargePlateAggregate", "8-ColumnAggregate"]
psd_list = ["DelanoeEtAl14", "FieldEtAl07TR", "FieldEtAl07ML"]


# %%
def cal_y_arts(ds_earthcare, habit_std, psd, coefs_exp=None):
    """Compute the brightness temperature for a given Earthcare dataset,
    habit and particle size distribution (PSD).
    Args:
        ds_earthcare (xarray.Dataset): Earthcare dataset with necessary fields.
        habit_std (str): Standard habit name to use.
        psd (str): Particle size distribution to use.
    Returns:
        xarray.DataArray: Brightness temperature in Planck brightness temperature units.
    """
    # Create a workspace
    ws = ea.wsInit(SpectralRegion.TIR)

    # Download ARTS catalogs if they are not already present.
    pyarts.cat.download.retrieve()

    # Use a customised scattering folder to combine PingYang data and liquid droplet
    ws.stdhabits_folder = "/scratch/li/arts-yang-liquid-simlink"

    # The main agenda for computations of the radiative transfer
    # follows a pure clears sky radiative transfer in this example.
    # When the path tracing hits the surface, the surface emission
    # is computed from the lowest atmospheric temperature,
    # and when the path tracing hits the top of the atmosphere,
    # the cosmic background is used as emission source.
    ws.iy_main_agendaSet(option="Emission")
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
    ws.surface_rtprop_agendaSet(option="Blackbody_SurfTFromt_field")
    ws.iy_space_agendaSet(option="CosmicBackground")

    # The path tracing is done step-by-step following a geometric path
    ws.ppath_agendaSet(option="FollowSensorLosPath")
    ws.ppath_step_agendaSet(option="GeometricPath")

    # We might have to compute the Jacobian for the retrieval
    # of relative humidity, so we need to set the agenda for
    # that.
    ws.water_p_eq_agendaSet(option="MK05")

    # The geometry of our planet, and several other properties, are
    # set to those of Earth.
    ws.PlanetSet(option="Earth")

    # Our output unit
    ws.iy_unit = "PlanckBT"  # Planck brightness temperature

    # We do not care about polarization
    ws.stokes_dim = 1

    # The atmosphere is assumed 1-dimensional
    ws.atmosphere_dim = 1

    # We have one sensor position in space, looking down, and one
    # sensor position at the surface, looking up.
    ws.sensor_pos = [[300e3]]
    ws.sensor_los = [[180]]

    # The dimensions of the problem are defined by a 1-dimensional pressure grid
    ws.p_grid = ds_earthcare["pressure"].data  # Pa
    ws.lat_grid = []
    ws.lon_grid = []

    # Read absorption lookup table
    # interporlation orders (defaults are too high)
    ws.abs_nls_interp_order = 3
    ws.abs_t_interp_order = 3
    ws.abs_p_interp_order = 3
    path_abs_lookup_table = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data/lookup_tables/abs_table_Earthcare_TIR2_2025-06-12_16:37:12.618561.xml",
    )
    ws.ReadXML(
        ws.abs_lookup,
        path_abs_lookup_table,
    )

    # Use the same absorption species and f_grid as in the lookup table
    # ws.f_gridFromGasAbsLookup()
    f_grid_abs_lookup = ws.abs_lookup.value.f_grid.value
    ws.f_grid = f_grid_abs_lookup[::5]  # use every 5th frequency point
    ws.abs_species = ws.abs_lookup.value.species

    # Compute the absorption agenda.
    ws.abs_lines_per_speciesSetEmpty()
    ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)
    ws.abs_lookupAdapt()

    # make atmosphere
    insert_atm_from_earthcare(ds_earthcare, ws)

    ws.jacobianOff()
    ws.sensorOff()

    # % Set the cloud layer
    scat_species = ["FWC", "LWC"]  # Arbitrary names
    insert_bulkprop_from_earthcare(ds_earthcare, ws, habit_std, psd, scat_species, coefs_exp=coefs_exp)

    # %
    # Cloudbox
    ea.scat_data(ws)
    ea.cloudbox(ws, CloudboxOption.AUTO)
    ws.pnd_fieldCalcFromParticleBulkProps()

    # We check the atmospheric geometry, the atmospheric fields, the cloud box,
    # and the sensor.
    ea.checks(ws)
    ea.disort(ws, Npfct=-1)

    # Switch off clouds?
    # ea.allsky2clearsky(ws)
    # print("All sky to clearsky done")

    # We perform the all sky calculations
    ws.yCalc()

    # save in data array
    return extract_xr_arrays(ds_earthcare, ws, scat_species)


def insert_bulkprop_from_earthcare(
    ds_earthcare, ws, habit_std, psd, scat_species=["FWC", "LWC"], coefs_exp=None
):
    for i, species in enumerate(scat_species):
        ws.Append(ws.particle_bulkprop_names, species)

        if species == "FWC":
            profile_fwc = ds_earthcare["frozen_water_content"].values
            wsv.particle_bulkprop_fieldInsert(ws, profile_fwc, i)
            ea.scat_data_rawAppendStdHabit(
                ws,
                habit=map_habit[habit_std],
                size_step=4,  # skip some particle sizes
            )

            # Set ice PSD
            if psd == "DelanoeEtAl14":
                ea.scat_speciesDelanoeEtAl14(ws, species)

            elif psd == "FieldEtAl07TR":
                ea.scat_speciesFieldEtAl07(ws, species, regime="TR")

            elif psd == "FieldEtAl07ML":
                ea.scat_speciesFieldEtAl07(ws, species, regime="ML")

            elif psd == "Exponential":
                if coefs_exp is None:
                    raise ValueError(
                        "Please provide the coefficients for the exponential PSD."
                    )
                n0 = coefs_exp.get("n0")
                ga = coefs_exp.get("ga", 1)
                ea.scat_speciesMgdMass(
                    ws, species, la=-999, mu=0, n0=n0, ga=ga, x_unit="dveq"
                )

            else:
                raise ValueError(f"PSD {psd} not handled")

        elif species == "LWC":
            profile_lwc = ds_earthcare["cloud_liquid_water_content"].values
            wsv.particle_bulkprop_fieldInsert(ws, profile_lwc, i)
            ea.scat_data_rawAppendStdHabit(
                ws,
                habit="MieSpheres_H2O_liquid",
                size_step=3,  # skip some particle sizes
                dmax_end=50e-6,  # ignore larger particles
            )

            coefs = get_mgd_coefs(remove_suffix=True)
            ea.scat_speciesMgdMass(ws, species, n0=-999, **coefs, x_unit="dveq")
        else:
            raise ValueError(f"Scattering species {species} not handled")


def get_mgd_coefs(remove_suffix=False):
    """
    Get the MGD coefficients for the parameters found in Tampieri & Tomasi (1975).
    Returns:
        mu_d (float): MGD mu parameter for diameter
        la_d (float): MGD lambda parameter for diameter
        ga_d (float): MGD gamma parameter for diameter
    """
    alpha = 5
    gamma = 1.49
    r_c = 8.81e-6  # m

    # change notation to match petty and Huang (2011)
    mu_r = alpha
    ga_r = gamma
    la_r = alpha / gamma / (r_c) ** (gamma)

    # convert to diameter parameters described in Petty and Huang (2011)
    mu_d = mu_r
    ga_d = ga_r
    la_d = la_r / 2**ga_d

    coefs = {"mu_d": mu_d, "la_d": la_d, "ga_d": ga_d}
    if remove_suffix:
        coefs = {k.split("_")[0]: v for k, v in coefs.items()}
    return coefs


def insert_atm_from_earthcare(ds_earthcare, ws):
    ws.z_field = ds_earthcare["height_grid"].data.reshape(-1, 1, 1)
    ws.t_field = ds_earthcare["temperature"].data.reshape(-1, 1, 1)
    if ds_earthcare["surfaceElevation"] > ds_earthcare["height_grid"].min():
        ws.z_surface = ds_earthcare["surfaceElevation"].data.reshape(1, 1)
    else:
        ws.z_surface = ds_earthcare["height_grid"].min().data.reshape(1, 1)

    vmr_h2o = ds_earthcare["h2o_volume_mixing_ratio"].data.reshape(-1, 1, 1)
    vmr_co2 = 427.53e-6 * np.ones(len(ws.z_field.value)).reshape(-1, 1, 1)
    vmr_n2o = 330e-9 * np.ones(len(ws.z_field.value)).reshape(-1, 1, 1)
    vmr_o3 = (
        28.9644
        / 47.9982
        * ds_earthcare["ozone_mass_mixing_ratio"].data.reshape(-1, 1, 1)
    )
    vmr_field_stack = []
    for s in ws.abs_species.value:
        s = str(s).split("-")[0]
        if s == "H2O":
            vmr_field_stack.append(vmr_h2o)
        elif s == "CO2":
            vmr_field_stack.append(vmr_co2)
        elif s == "N2O":
            vmr_field_stack.append(vmr_n2o)
        elif s == "O3":
            vmr_field_stack.append(vmr_o3)
        else:
            raise (f"Absorption species {s} is not implemented")
    ws.vmr_field = np.stack(vmr_field_stack)
    return


def extract_xr_arrays(ds_earthcare, ws, scat_species):
    da_y = xr.DataArray(
        name="IR Temperature",
        data=ws.y.value,
        coords={
            "f_grid": pyarts.arts.convert.freq2wavelen(ws.f_grid.value.value) * 1e6
        },
        attrs={
            "unit": "K",
            "long_name": "Brightness Temperature",
        },
    ).sortby("f_grid")

    da_bulkprop_field = xr.DataArray(
        name="particle_bulkprob_field",
        data=ws.particle_bulkprop_field.value.value.squeeze(),
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


def get_frozen_water_content(ds_onion_invtable, ds_earthcare_):
    lowest_dBZ_threshold = -30
    ds_earthcare_["dBZ"] = ds_earthcare_["dBZ"].where(
        ds_earthcare_["dBZ"] >= lowest_dBZ_threshold
    )

    profile_fwc = (
        ds_onion_invtable.sel(radiative_properties="FWC")
        .interp(
            Temperature=ds_earthcare_["temperature"],
            dBZ=ds_earthcare_["dBZ"],
        )
        .pipe(lambda x: 10**x)  # Convert from log10(FWC) to FWC in kg/m^3
        .fillna(0)
    )

    ds_earthcare_ = ds_earthcare_.assign(
        frozen_water_content=(
            ds_earthcare_.dims,
            profile_fwc.where(
                profile_fwc >= 0,
                0,  # ensure no negative values
            ).data,
            dict(
                long_name="Frozen Water Content",
                units="kg m-3",
            ),
        )
    )

    return ds_earthcare_


def get_cloud_top_height(ds, fwc_threshold=1e-5):
    """Calculate the cloud top height based on the frozen water content."""
    ds["cloud_top_height"] = (
        ds["height_grid"]
        .where(ds["frozen_water_content"] > fwc_threshold)
        .max("height_grid")
    ).assign_attrs(
        long_name="Cloud Top Height",
        units="m",
        description="Height at which the frozen water content exceeds the threshold",
        fwc_threshold=fwc_threshold,
    )
    return ds


def get_cloud_top_T(ds, fwc_threshold=1e-5):
    ds = get_cloud_top_height(ds, fwc_threshold=fwc_threshold)
    ds["cloud_top_T"] = (
        ds["temperature"]
        .where(ds["height_grid"] == ds["cloud_top_height"])
        .min("height_grid", skipna=True)
    ).assign_attrs(
        long_name="Cloud Top Temperature",
        units="K",
        description="Temperature at the cloud top height",
        fwc_threshold=fwc_threshold,
    )
    return ds


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
        f"data/earthcare/arts_output_data/cold_3rd_{habit_std}_{psd}_{orbit_frame}.nc",
    )

    # %% check if the file already exists
    if os.path.exists(file_save_ncdf):
        print(f"File {file_save_ncdf} already exists. Skipping computation.")
        sys.exit(0)

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

    # remove clearsky profiles
    mask_low_fwc = (
        ds_earthcare_["frozen_water_content"].sel(height_grid=slice(5e3, None)) == 0
    ).all(dim="height_grid")
    mask_cold_clouds = ds_earthcare_["pixel_values"] < 245  # K
    mask = np.logical_and(~mask_low_fwc, mask_cold_clouds)

    if (~mask).all():
        print("All nrays are cleared sky. No computation needed.")
        sys.exit(0)

    ds_earthcare_subset = ds_earthcare_.where(mask, drop=True).isel(
        nray=slice(None, None, 3),  # skip every xth ray to reduce computation time
    )
    print(f"Number of nrays: {len(ds_earthcare_subset.nray)}")

    # %%
    # % loop over nrays
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

    with ProcessPoolExecutor(max_workers=64) as executor:
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
    if len(ds_arts.nray) > 1:
        ds_arts = ds_arts.sortby("time")

    ds_arts = xr.merge([ds_arts, ds_earthcare_subset])

    # %%
    # %% save to file
    ds_arts.to_netcdf(file_save_ncdf)
