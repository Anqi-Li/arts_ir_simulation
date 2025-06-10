# %%
import pyarts
import numpy as np
import os
import xarray as xr
import time
import easy_arts.profiles as prof
import easy_arts.wsv as wsv
import easy_arts.easy_arts as ea
from easy_arts.data_model import (
    CloudboxOption,
    SpectralRegion,
)
import copy
import sys
sys.path.append(os.path.dirname(os.getcwd()))


# %%
# Create a workspace
ws = ea.wsInit(SpectralRegion.TIR)

# Download ARTS catalogs if they are not already present.
pyarts.cat.download.retrieve()

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
ws.p_grid = np.logspace(np.log10(1050e2), np.log10(10e2), 150)  # Pa
ws.lat_grid = []
ws.lon_grid = []

# The surface is at 0-meters altitude
ws.z_surface = [[0.0]]

# Read lookup table
# interporlation orders (defaults are too high)
ws.abs_nls_interp_order = 3
ws.abs_t_interp_order = 3
ws.abs_p_interp_order = 3
ws.ReadXML(
    ws.abs_lookup,
    os.path.join(
        os.getcwd(),
        "data/lookup_table_perturb14_2025-03-13_11:29:32.528647.xml",
    ),
)
# The lookup table contains the absorption lines for the
# gases in the atmosphere, and the frequencies at which
# they are computed.  We can use this to set the frequency
# grid for the absorption lines.
# ws.f_gridFromGasAbsLookup()
f_grid_abs_lookup = ws.abs_lookup.value.f_grid.value
ws.f_grid = f_grid_abs_lookup[1::10]

# The atmosphere consists of water, oxygen and nitrogen.
# We set these to be computed using predefined absorption
# models.
ws.abs_species = ws.abs_lookup.value.species

# We now have all the information required to compute the absorption agenda.
ws.abs_lines_per_speciesSetEmpty()
ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)
ws.abs_lookupAdapt()

# We need an atmosphere.  This is taken from the arts-xml-data in
# raw format before being turned into the atmospheric fields that
# ARTS uses internally.
ws.AtmRawRead(basename="planets/Earth/Fascod/midlatitude-winter/midlatitude-winter")
ws.AtmFieldsCalc()

# Set the cloud layer
scat_species = ["IWC"]
habit = "5-PlateAggregate-Smooth"

IWP = 0.1  # kg/m2
fwhm = 1e3  # km
z_ref = 8e3  # km
# Create particle_bulkprop_field/names
z = ws.z_field.value[:, 0, 0]
z0 = ws.z_surface.value[0, 0]
t = ws.t_field.value[:, 0, 0]
for i, species in enumerate(scat_species):
    ws.Append(ws.particle_bulkprop_names, species)

    if species == "IWC":
        # A cropped Gaussian SWC profile
        profile = prof.gaussian(
            z=z,
            y_at_ref=1,  # Dummy value, as normalisation done below
            fwhm=fwhm,  
            z_ref=z_ref,  
            z_max=11e3,
            t=t,
            t_min=210,
            t_max=274,
        )
        prof.norm_integral(profile, z, IWP, z0)
        wsv.particle_bulkprop_fieldInsert(ws, profile, i)

    else:
        raise ValueError(f"Scattering species {species} not handled")

# Define scattering species
for species in ws.particle_bulkprop_names.value:

    if species == "IWC":
        ea.scat_speciesMgdMass(ws, species)
        ea.scat_data_rawAppendStdHabit(ws, habit=habit, size_step=4)

    else:
        raise ValueError(f"Scattering species {species} not handled")

# These calculations do no partial derivatives, so we can turn it off
ws.jacobianOff()

# Cloudbox
ea.scat_data(ws)
ea.cloudbox(ws, CloudboxOption.AUTO)
ws.pnd_fieldCalcFromParticleBulkProps()


# The concept of a sensor does not apply to this example, so we can turn it off
ws.sensorOff()

# We check the atmospheric geometry, the atmospheric fields, the cloud box,
# and the sensor.
ea.checks(ws)
ea.disort(ws, Npfct=-1)

# We perform the all sky calculations
start = time.time()
ws.yCalc()
end = time.time()
time_delta = end - start
print(time_delta, "s")
print(habit)
y_allsky = copy.deepcopy(ws.y.value)

# save in data array
da = xr.DataArray(
    name=habit,
    data=y_allsky,
    coords={"f_grid": pyarts.arts.convert.freq2kaycm(ws.f_grid.value.value)},
    attrs={
        "unit": "K",
        "iwp": IWP,
        "z_ref": z_ref,
        "fwhm": fwhm,
    },
)
#
da.to_netcdf(
    os.path.join(
        os.getcwd(),
        "data/allsky_habits_MgdMass/y_{}.nc".format(
            habit,
            # time.strftime("%Y%m%d-%H:%M:%S", time.localtime()),
        ),
    ),
)

# %% clear sky calculations
ea.allsky2clearsky(ws)
start = time.time()
ws.yCalc()
end = time.time()
time_delta = end - start
print("clearsky", time_delta, "s")
y_clearsky = copy.deepcopy(ws.y.value)

# %% organised into dataset
ds = xr.Dataset(
    data_vars={
        "allsky": ("f_grid", y_allsky),
        "clearsky": ("f_grid", y_clearsky),
    },
    coords={"f_grid": pyarts.arts.convert.freq2kaycm(ws.f_grid.value.value)},
)
