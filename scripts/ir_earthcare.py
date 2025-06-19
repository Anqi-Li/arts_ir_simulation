# %%
import pyarts
import numpy as np
import os
import xarray as xr
from datetime import datetime
from tqdm import tqdm
import easy_arts.wsv as wsv
import easy_arts.easy_arts as ea
from easy_arts.data_model import (
    CloudboxOption,
    SpectralRegion,
)
from physics.unitconv import specific_humidity2h2o_p
import matplotlib.pyplot as plt

path_abs_lookup_table = os.path.join(
    os.getcwd(),
    "../data/lookup_tables/abs_table_Earthcare_TIR2_2025-06-12_16:37:12.618561.xml",
)

# % map the standard habit to the Yang habit name
map_habit = {
    "LargePlateAggregate": "Plate-Smooth",
    "8-ColumnAggregate": "8-ColumnAggregate-Smooth",
}


# %%
def cal_y_arts(ds_earthcare, habit_std, psd, ds_onion_invtable):
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
    ws.iy_unit = "PlanckBT"  # "1"# W/(m^2 Hz sr)

    # We do not care about polarization in this example
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
    ws.ReadXML(
        ws.abs_lookup,
        path_abs_lookup_table,
    )

    # Use the same absorption species and f_grid as in the lookup table
    # ws.f_gridFromGasAbsLookup()
    f_grid_abs_lookup = ws.abs_lookup.value.f_grid.value
    ws.f_grid = f_grid_abs_lookup[1::1]
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
    insert_bulkprop_from_earthcare(
        ds_earthcare, ws, habit_std, psd, ds_onion_invtable, scat_species
    )

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

    # We perform the all sky calculations
    ws.yCalc()

    # save in data array
    return extract_xr_arrays(ds_earthcare, ws, scat_species)


def insert_bulkprop_from_earthcare(
    ds_earthcare, ws, habit_std, psd, ds_onion_invtable, scat_species=["FWC", "LWC"]
):
    for i, species in enumerate(scat_species):
        ws.Append(ws.particle_bulkprop_names, species)

        if species == "FWC":
            # Create profile from earthcare radar by using the onion inversion table
            profile_earthcare_fwc_log10 = ds_onion_invtable.sel(
                radiative_properties="FWC"
            ).interp(
                dBZ=ds_earthcare["dBZ"],
                Temperature=ds_earthcare["temperature"],
                method="linear",
            )

            profile_earthcare_fwc = profile_earthcare_fwc_log10.pipe(
                lambda x: 10**x
            )  # Convert from log10(FWC) to FWC in kg/m^3
            profile_fwc = profile_earthcare_fwc.fillna(0).values
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

            else:
                raise ValueError(f"PSD {psd} not handled")

        elif species == "LWC":
            profile_lwc = ds_earthcare["cloud_liquid_water_content"].fillna(0).values
            wsv.particle_bulkprop_fieldInsert(ws, profile_lwc, i)
            ea.scat_data_rawAppendStdHabit(
                ws,
                habit="MieSpheres_H2O_liquid",
                size_step=3,  # skip some particle sizes
                dmax_end=50e-3,  # ignore larger particles
            )
            ea.scat_speciesMgdMass(ws, species)

        else:
            raise ValueError(f"Scattering species {species} not handled")


def insert_atm_from_earthcare(ds_earthcare, ws):
    ws.z_field = ds_earthcare["height_grid"].data.reshape(-1, 1, 1)
    ws.t_field = ds_earthcare["temperature"].data.reshape(-1, 1, 1)
    if ds_earthcare["surfaceElevation"] > ds_earthcare["height_grid"].min():
        ws.z_surface = ds_earthcare["surfaceElevation"].data.reshape(1, 1)
    else:
        ws.z_surface = ds_earthcare["height_grid"].min().data.reshape(1, 1)

    vmr_h2o = (
        (
            specific_humidity2h2o_p(
                ds_earthcare["specific_humidity"],
                ds_earthcare["pressure"],
            )
            / ds_earthcare["pressure"]
        )
        .data
        .reshape(-1, 1, 1)
    )

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
        # data=profile_fwc,
        data=ws.particle_bulkprop_field.value.value.squeeze(),
        coords={
            "scat_species": scat_species,
            "height_grid": ds_earthcare["height_grid"],
        },
        attrs={
            "unit": "kg m-3",
            # "long_name": "frozen water content",
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

    return da_y, da_bulkprop_field, da_vmr_field


# %% choose invtable
habit_std = ["LargePlateAggregate", "8-ColumnAggregate"][0]  # Habit to use
psd = ["DelanoeEtAl14", "FieldEtAl07TR", "FieldEtAl07ML"][0]  # PSD to use
print(habit_std)
print(psd)

ds_onion_invtable = xr.open_dataset(
    os.path.join(
        os.getcwd(),
        f"../data/onion_invtables/onion_invtable_{habit_std}_{psd}.nc",
    ),
)[f"onion_invtable_{habit_std}_{psd}"]


# %% Load Earthcare data
# take a sample earthcare dataset
path_earthcare = "../data/earthcare/arts_x_data/"
orbit_frame = "01162E"
ds_earthcare_ = xr.open_dataset(path_earthcare + f"arts_x_{orbit_frame}.nc")

# make LWC
ds_earthcare_ = ds_earthcare_.eval("density = pressure/(temperature * 287e3)")
ds_earthcare_["density"] = ds_earthcare_["density"].assign_attrs(
    dict(
        long_name="Air Density",
        units="kg m-3",
    )
)
ds_earthcare_ = ds_earthcare_.eval(
    "cloud_liquid_water_content = specific_cloud_liquid_water_content * density"
)
ds_earthcare_["cloud_liquid_water_content"] = ds_earthcare_[
    "cloud_liquid_water_content"
].assign_attrs(
    dict(
        long_name="Liquid Water Content",
        units="kg m-3",
    )
)
# fill nan and inf
ds_earthcare_["dBZ"] = ds_earthcare_["dBZ"].fillna(-50)
ds_earthcare_["dBZ"] = ds_earthcare_["dBZ"].where(~np.isinf(ds_earthcare_["dBZ"]), -50)

# remove low cloud profiles
low_dBZ_threshold = -35
mask_low_cloud = (
    ds_earthcare_.sel(height_grid=slice(0, None))["dBZ"] < low_dBZ_threshold
).all(dim="height_grid")
mask_low_lwc = (ds_earthcare_["cloud_liquid_water_content"] == 0).all(dim="height_grid")
mask = np.logical_and(mask_low_cloud, mask_low_lwc)
# ds_earthcare_ = ds_earthcare_.where(~mask_low_cloud, drop=True)


# %% loop over nrays
ds_earthcare_subset = ds_earthcare_.where(mask_low_cloud, drop=True).isel(
    # nray=np.linspace(0, 5000, 10, dtype=int)
    nray=slice(None, 29, 2)
)
ds_earthcare_subset = ds_earthcare_subset.interpolate_na(dim='height_grid',fill_value="extrapolate")
y = []
# fwc = []
bulkprop = []
vmr = []
for i in tqdm(range(len(ds_earthcare_subset.nray)), desc="process nrays"):
    da_y, da_bulkprop, da_vmr = cal_y_arts(
        ds_earthcare_subset.isel(nray=i),
        habit_std,
        psd,
        ds_onion_invtable,
    )
    y.append(da_y)
    bulkprop.append(da_bulkprop)
    vmr.append(da_vmr)
    # fwc.append(da_fwc)

da_y = xr.concat(y, dim="nray")
da_bulkprop = xr.concat(bulkprop, dim="nray")
da_vmr = xr.concat(vmr, dim="nray")
# da_fwc = xr.concat(fwc, dim="nray")

# %% Plotting
plt.figure()
da_bulkprop.plot(
    y="height_grid", hue="nray", col="scat_species", sharex=False, xscale="log"
)
plt.show()

plt.figure()
da_vmr.plot(y="height_grid", hue="nray", col="abs_species", sharex=False, xscale="log")
plt.show()

plt.figure()
kwargs = dict(marker=".")
da_y.isel(f_grid=slice(None, None, 10)).mean("f_grid").plot(x="nray", label="arts 10th", **kwargs)
da_y.isel(f_grid=slice(None, None, 5)).mean("f_grid").plot(x="nray", label="arts 5th", **kwargs)
da_y.isel(f_grid=slice(None, None, 1)).mean("f_grid").plot(x="nray", label="arts 1th", **kwargs)

ds_earthcare_subset["pixel_values"].plot(x="nray", label="MSI", **kwargs)
ds_earthcare_subset["temperature"].max(dim="height_grid").plot(
    x="nray", label="max T", **kwargs
)
ds_earthcare_subset.temperature.interp(
    height_grid=ds_earthcare_subset.surfaceElevation,
    kwargs={"fill_value": "extrapolate"},
).plot(x='nray', label='surface T (extrapolate)', **kwargs)
plt.ylim([286, 302])
plt.legend()
plt.title(
    f"""
{habit_std}
{psd}
{orbit_frame}
          """
)
plt.show()
# %% save to file
ds = ds_earthcare_.assign(
    y_arts=da_y,
    # fwc=da_fwc,
    bulkprop=da_bulkprop,
    vmr=da_vmr,
)
# ds.to_netcdf(f"../data/earthcare/arts_y_data/y_lwc_{habit_std}_{psd}_{orbit_frame}.nc")


