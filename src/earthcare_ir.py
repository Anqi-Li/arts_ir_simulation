# %%
import sys
import glob
import pyarts
import numpy as np
from datetime import datetime
import os
import xarray as xr
from tqdm import tqdm
import easy_arts.wsv as wsv
import easy_arts.easy_arts as ea
from easy_arts.data_model import (
    CloudboxOption,
    SpectralRegion,
)
from physics.constants import DENSITY_H2O_LIQUID
from concurrent.futures import ProcessPoolExecutor, as_completed
from onion_table import *
from physics.unitconv import (
    mixing_ratio2mass_conc,
    specific_humidity2h2o_p,
)
from ectools import ecio
import data_paths

# % map the standard habit to the Yang habit name
map_habit = {
    "LargePlateAggregate": "Plate-Smooth",
    "8-ColumnAggregate": "8-ColumnAggregate-Smooth",
    "6-BulletRosette": "SolidBulletRosette-Smooth",
}

# deprecated! Use classes below instead!
habit_std_list = ["LargePlateAggregate", "8-ColumnAggregate", "6-BulletRosette"]
psd_list = ["DelanoeEtAl14", "FieldEtAl07TR", "FieldEtAl07ML", "ModifiedGamma"]


class Habit:
    Plate = "LargePlateAggregate"
    Column = "8-ColumnAggregate"
    Bullet = "6-BulletRosette"


class PSD:
    D14 = "DelanoeEtAl14"
    F07T = "FieldEtAl07TR"
    F07M = "FieldEtAl07ML"
    MDG = "ModifiedGamma"


# %%
def cal_y_arts(ds_earthcare, habit_std, psd, coef_mgd=None):
    """Compute the brightness temperature for a given Earthcare dataset,
    habit and particle size distribution (PSD).
    Args:
        ds_earthcare (xarray.Dataset): Earthcare dataset with necessary fields.
        habit_std (str): Standard habit name to use.
        psd (str): Particle size distribution to use.
    Returns:
        xarray.DataArray: Brightness temperature in Planck brightness temperature units.
    """

    # Shorten the input profile where pressure grid is NaN
    ds_earthcare_org = ds_earthcare.copy()
    ds_earthcare = ds_earthcare_org.where(ds_earthcare_org["pressure"].notnull(), drop=True)
    if len(ds_earthcare["CPR_height"]) < 2:
        raise ValueError("Not enough valid pressure levels in the profile.")

    # Create a workspace
    ws = ea.wsInit(SpectralRegion.TIR)
    ws.SetNumberOfThreads(nthreads=1)

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
    insert_bulkprop_from_earthcare(ds_earthcare, ws, habit_std, psd, scat_species, coef_mgd=coef_mgd)

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
    ds_earthcare,
    ws,
    habit_std,
    psd,
    scat_species=["FWC", "LWC"],
    coef_mgd=None,
    habit_pingyang=True,
):
    for i, species in enumerate(scat_species):
        ws.Append(ws.particle_bulkprop_names, species)

        if species == "FWC":
            profile_fwc = ds_earthcare["frozen_water_content"].values
            wsv.particle_bulkprop_fieldInsert(ws, profile_fwc, i)
            ea.scat_data_rawAppendStdHabit(
                ws,
                habit=map_habit[habit_std] if habit_pingyang else habit_std,
                size_step=4,  # skip some particle sizes
            )

            # Set ice PSD
            if psd == "DelanoeEtAl14":
                ea.scat_speciesDelanoeEtAl14(ws, species)

            elif psd == "FieldEtAl07TR":
                ea.scat_speciesFieldEtAl07(ws, species, regime="TR")

            elif psd == "FieldEtAl07ML":
                ea.scat_speciesFieldEtAl07(ws, species, regime="ML")

            elif psd == "ModifiedGamma":
                if coef_mgd is None:
                    raise ValueError("Please provide the coefficients for the Modified Gamma PSD.")
                n0 = coef_mgd.get("n0")
                ga = coef_mgd.get("ga", 1)
                mu = coef_mgd.get("mu", 0)
                ea.scat_speciesMgdMass(ws, species, la=-999, mu=mu, n0=n0, ga=ga, x_unit="dveq")

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
    ws.z_field = ds_earthcare["height"].data.reshape(-1, 1, 1)
    ws.t_field = ds_earthcare["temperature"].data.reshape(-1, 1, 1)
    if ds_earthcare["surface_elevation"].min() > ds_earthcare["height"].min():
        ws.z_surface = ds_earthcare["surface_elevation"].min().data.reshape(1, 1)
    else:
        ws.z_surface = ds_earthcare["height"].min().data.reshape(1, 1)

    vmr_field_stack = []
    for s in ws.abs_species.value:
        s = str(s).split("-")[0]
        if s == "H2O":
            vmr_h2o = ds_earthcare["h2o_volume_mixing_ratio"].data.reshape(-1, 1, 1)
            vmr_field_stack.append(vmr_h2o)
        elif s == "CO2":
            vmr_co2 = 427.53e-6 * np.ones(len(ws.z_field.value)).reshape(-1, 1, 1)
            vmr_field_stack.append(vmr_co2)
        elif s == "N2O":
            vmr_n2o = 330e-9 * np.ones(len(ws.z_field.value)).reshape(-1, 1, 1)
            vmr_field_stack.append(vmr_n2o)
        elif s == "O3":
            vmr_o3 = 28.9644 / 47.9982 * ds_earthcare["ozone_mass_mixing_ratio"].fillna(0).data.reshape(-1, 1, 1)
            vmr_field_stack.append(vmr_o3)
        elif s == "O2":
            vmr_o2 = 0.2095 * np.ones(len(ws.z_field.value)).reshape(-1, 1, 1)
            vmr_field_stack.append(vmr_o2)
        elif s == "N2":
            vmr_n2 = 0.7809 * np.ones(len(ws.z_field.value)).reshape(-1, 1, 1)
            vmr_field_stack.append(vmr_n2)
        elif s == "liquidcloud":
            vmr_lwc = ds_earthcare["cloud_liquid_water_content"].data.reshape(-1, 1, 1) / DENSITY_H2O_LIQUID
            vmr_field_stack.append(vmr_lwc)
        else:
            print(s)
            raise (f"Absorption species {s} is not implemented")
    ws.vmr_field = np.stack(vmr_field_stack)
    return


def extract_xr_arrays(ds_earthcare, ws, scat_species):
    da_y = xr.DataArray(
        name="IR Temperature",
        data=ws.y.value,
        coords={"f_grid": pyarts.arts.convert.freq2wavelen(ws.f_grid.value.value) * 1e6},
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
            "height": ds_earthcare["height"].data,
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
            "height": ds_earthcare["height"].data,
        },
        attrs={
            "unit": "1",
            "long_name": "absoption species vmr",
        },
    )
    da_auxiliary = ds_earthcare[["time", "latitude", "longitude"]].copy()

    return da_y, da_bulkprop_field, da_vmr_field, da_auxiliary


def get_frozen_water_content(ds_onion_invtable, ds_earthcare):
    lowest_dBZ_threshold = -30
    ds_earthcare["reflectivity_corrected"] = ds_earthcare["reflectivity_corrected"].where(
        ds_earthcare["reflectivity_corrected"] >= lowest_dBZ_threshold
    )

    profile_fwc = (
        ds_onion_invtable.sel(radiative_properties="FWC")
        .interp(
            Temperature=ds_earthcare["temperature"],
            dBZ=ds_earthcare["reflectivity_corrected"],
        )
        .pipe(lambda x: 10**x)  # Convert from log10(FWC) to FWC in kg/m^3
        .fillna(0)
    )

    ds_earthcare = ds_earthcare.assign(
        frozen_water_content=(
            ds_earthcare.dims,
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

    return ds_earthcare


def get_frozen_water_path(ds):
    """Calculate the frozen water path from the frozen water content."""
    ds["frozen_water_path"] = (
        ds["frozen_water_content"]
        .integrate("height_grid")
        .assign_attrs(
            long_name="Frozen Water Path",
            units="kg/m^2",
            description="Integrated frozen water content over height grid",
        )
    )
    return ds


def get_cloud_top_height(ds, fwc_threshold=1e-5, dbz_threshold=-30, based_on_fwc=True):
    """Calculate the cloud top height based on the frozen water content (default), otherwise based on dBZ."""
    if based_on_fwc:
        ds["cloud_top_height"] = (ds["height_grid"].where(ds["frozen_water_content"] > fwc_threshold).max("height_grid")).assign_attrs(
            long_name="Cloud Top Height",
            units="m",
            description="Height at which the frozen water content exceeds the threshold",
            fwc_threshold=fwc_threshold,
        )
    else:
        ds["cloud_top_height"] = (ds["height_grid"].where(ds["reflectivity_corrected"] > dbz_threshold).max("height_grid")).assign_attrs(
            long_name="Cloud Top Height",
            units="m",
            description="Height at which the dBZ exceeds the threshold",
            dbz_threshold=dbz_threshold,
        )
    return ds


def get_cloud_top_T(ds, fwc_threshold=None, dbz_threshold=None, based_on_fwc=True):
    ds = get_cloud_top_height(
        ds,
        fwc_threshold=fwc_threshold,
        dbz_threshold=dbz_threshold,
        based_on_fwc=based_on_fwc,
    )
    ds["cloud_top_T"] = (ds["temperature"].where(ds["height_grid"] == ds["cloud_top_height"]).min("height_grid", skipna=True)).assign_attrs(
        long_name="Cloud Top Temperature",
        units="K",
        description="Temperature at the cloud top height",
        fwc_threshold=fwc_threshold,
    )
    return ds


def get_lwc_and_h2o_vmr(ds_earthcare):
    # make H2O VMR from specific_humidity
    ds_earthcare["h2o_volume_mixing_ratio"] = (
        (
            specific_humidity2h2o_p(
                ds_earthcare["specific_humidity"],
                ds_earthcare["pressure"],
            )
            / ds_earthcare["pressure"]
        )
        .fillna(0)
        .assign_attrs(
            dict(
                long_name="Water Vapor Volume Mixing Ratio",
                units="kg kg-1",
            )
        )
    )

    ds_earthcare["cloud_liquid_water_content"] = mixing_ratio2mass_conc(
        ds_earthcare["specific_cloud_liquid_water_content"],  # kg/kg
        ds_earthcare["pressure"],  # Pa
        ds_earthcare["temperature"],  # K
    ).assign_attrs(
        dict(
            long_name="Cloud Liquid Water Content",
            units="kg m-3",
        )
    )

    ds_earthcare["cloud_liquid_water_content"] = ds_earthcare["cloud_liquid_water_content"].where(
        ds_earthcare["cloud_liquid_water_content"] >= 0,
        0,  # ensure no negative values
    )
    return ds_earthcare


# %%
def get_inputs(orbit_frame: str, skip_profiles: int = 5, low_reflectivity_threshold: float = -15):
    ds_xmet = ecio.load_XMET(
            srcpath=data_paths.XMET,
            frame_code=orbit_frame,
            nested_directory_structure=True,
        )
    ds_xmet.close()
    ds_cfmr = ecio.get_XMET(
        XMET=ds_xmet,
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

    ds_cfmr.close()
    print("C-FMR and X-MET loading done.")

    # Remove profiles with all NaN or very low reflectivity
    nan_pressure = ds_cfmr["pressure"].isnull().all(dim="CPR_height")
    nan_height = ds_cfmr["height"].isnull().all(dim="CPR_height")
    low_reflectivity = (ds_cfmr["reflectivity_corrected"].fillna(-999) < low_reflectivity_threshold).all(dim="CPR_height")
    mask = ~(nan_pressure | nan_height | low_reflectivity)

    ds_cfmr_subset = ds_cfmr.where(mask, drop=True).isel(
        along_track=slice(None, None, skip_profiles),  # skip every xth ray to reduce computation time
    )

    # reverse the height dimension so that pressure is increasing
    ds_cfmr_subset = ds_cfmr_subset.isel(CPR_height=slice(None, None, -1))
    ds_cfmr_subset.encoding = ds_cfmr.encoding  # keep the original encoding info

    print(f"Subset number of profiles: {len(ds_cfmr_subset.along_track)}")

    # Prepare h2o_volume_mixing_ratio and liquid_water_content
    ds_cfmr_subset = get_lwc_and_h2o_vmr(ds_cfmr_subset)
    print("LWC and H2O VMR preparation done.")
    return ds_cfmr_subset


def main(
    orbit_frame: str,
    skip_profiles: int = 5,
    habit_list: list = [Habit.Bullet, Habit.Column, Habit.Plate],
    psd_list: list = [PSD.D14, PSD.MDG, PSD.F07T],
    low_reflectivity_threshold: float = -15,
    skip_existing: bool = True,
    max_workers: int = 32,
    save_results: bool = False,
):
    print(f"Processing orbit frame: {orbit_frame}")

    # check if all habits and psds for this orbit frame have already been computed
    existed_files = 0
    for habit in habit_list:
        for psd in psd_list:
            filename_save_ncdf = os.path.join(
                data_paths.arts_output_TIR2,
                f"arts_TIR2_{orbit_frame}_{habit}_{psd}.nc",
            )
            if os.path.exists(filename_save_ncdf):
                existed_files += 1
    if existed_files == len(habit_list) * len(psd_list) and skip_existing:
        print(f"All combinations of habit and PSD for orbit frame {orbit_frame} already exist. Skipping computation.")
        return None

    # Load Earthcare data
    ds_cfmr_subset = get_inputs(orbit_frame, skip_profiles, low_reflectivity_threshold)

    for habit in habit_list:
        for psd in psd_list:
            print(f"Selected habit: {habit}, PSD: {psd}")
            filename_save_ncdf = os.path.join(
                data_paths.arts_output_TIR2,
                f"arts_TIR2_{orbit_frame}_{habit}_{psd}.nc",
            )
            if os.path.exists(filename_save_ncdf) & skip_existing:
                print(f"File {filename_save_ncdf} already exists. Skipping computation.")
                continue

            coef_mgd = (
                {
                    "n0": 1e10,  # Number concentration
                    "ga": 1.5,  # Gamma parameter
                    "mu": 0,  # Default value for mu
                }
                if psd == PSD.MDG
                else None
            )

            # Invertion to get frozen water content
            ds_cfmr_subset = get_frozen_water_content(
                get_ds_onion_invtable(habit=habit, psd=psd, coef_mgd=coef_mgd),
                ds_cfmr_subset,
            )
            print("FWC inversion done.")

            # ARTS simulation for each profile
            y = []
            auxiliary = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        cal_y_arts,
                        ds_cfmr_subset.isel(along_track=i),
                        habit,
                        psd,
                        coef_mgd,
                    )
                    for i in range(len(ds_cfmr_subset.along_track))
                ]
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

            # concatenate results
            da_y = xr.concat(y, dim="along_track")
            da_auxiliary = xr.concat(auxiliary, dim="along_track")

            ds_arts = da_auxiliary.assign({"arts": da_y.mean("f_grid")})
            ds_arts["arts"].attrs.update(
                {
                    "long_name": "ARTS simulated brightness temperature",
                    "units": "K",
                    "CPR source": ds_cfmr_subset.encoding["source"].split("/")[-1],
                    "habit": habit,
                    "PSD": psd,
                    "coef_mgd": str(coef_mgd) if coef_mgd is not None else "None",
                }
            )
            if len(ds_arts.along_track) > 1:
                ds_arts = ds_arts.sortby("along_track")
            print("Concatenating results done.")

            if save_results:
                ds_arts.to_netcdf(filename_save_ncdf)
                print(f"ARTS result saved to {filename_save_ncdf}")

    return ds_arts


# %%
if __name__ == "__main__":
    filelist_CFMR = ecio.get_filelist(
        srcpath=os.path.join(data_paths.CFMR, "*", "*", "*"),
        prodmod_code="ECA_EXBA",
    )
    filelist_XMET = ecio.get_filelist(
        srcpath=os.path.join(data_paths.XMET, "*", "*", "*"),
        prodmod_code="ECA_EXAA",
    )
    filelist_MRGR = ecio.get_filelist(
        srcpath=os.path.join(data_paths.MRGR, "*", "*", "*"),
        prodmod_code="ECA_EXBA",
    )
    orbit_frame_list_CFMR = [f.split("/")[-1].split("_")[-1].split(".")[0] for f in filelist_CFMR]
    orbit_frame_list_XMET = [f.split("/")[-1].split("_")[-1].split(".")[0] for f in filelist_XMET]
    orbit_frame_list_MRGR = [f.split("/")[-1].split("_")[-1].split(".")[0] for f in filelist_MRGR]
    common_orbit_frame_list = list(set(orbit_frame_list_CFMR) & set(orbit_frame_list_XMET) & set(orbit_frame_list_MRGR))

    # %% set up logging
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "log",
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create a new log file with timestamp
    log_path = os.path.join(log_dir, f"earthcare_ir_{datetime.now():%Y%m%d_%H%M%S}.log")

    # Keep only the 5 most recent log files
    log_files = sorted(glob.glob(os.path.join(log_dir, "earthcare_ir_*.log")))
    while len(log_files) > 4:  # will become 5 after this run
        os.remove(log_files[0])
        log_files = sorted(glob.glob(os.path.join(log_dir, "earthcare_ir_*.log")))

    # %% process all common orbit frames
    for orbit_frame in sorted(common_orbit_frame_list):
        # catch errors and continue
        try:
            _ = main(
                orbit_frame=orbit_frame,
                skip_profiles=5,
                habit_list=[Habit.Bullet, Habit.Column, Habit.Plate],
                psd_list=[PSD.D14, PSD.MDG, PSD.F07T],
                skip_existing=True,
                max_workers=32,
                save_results=True,
            )
            print(f"Finished orbit frame: {orbit_frame}")
        except Exception as e:
            print(f"Error processing orbit frame {orbit_frame}: {e}")
            # write error to log file
            with open(log_path, "a") as logfile:
                logfile.write(f"Error processing orbit frame {orbit_frame}: {e}\n")
            continue
