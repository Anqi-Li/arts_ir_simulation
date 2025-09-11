# Import the module
import pyarts
import numpy as np  # This example uses numpy
import matplotlib.pyplot as plt
import easy_arts.wsv as wsv
from easy_arts.data_model import (
    FascodVersion,
    SpectralRegion,
)
import easy_arts.easy_arts as ea


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

# Our output unit is Planc brightness temperature
ws.iy_unit = "1"  # W/(m^2 Hz sr)

# We do not care about polarization in this example
ws.stokes_dim = 1

# The atmosphere is assumed 1-dimensional
ws.atmosphere_dim = 1

# We have one sensor position in space, looking down, and one
# sensor position at the surface, looking up.
ws.sensor_pos = [[300e3]]
ws.sensor_los = [[180]]

# The dimensions of the problem are defined by a 1-dimensional pressure grid
ws.p_grid = np.logspace(np.log10(1050e2), np.log10(100), 100)  # Pa
ws.lat_grid = []
ws.lon_grid = []

# The surface is at 0-meters altitude
ws.z_surface = [[0.0]]

# Our sensor's frequency bins
f_grid_kayser = np.linspace(500, 1250, 500)  # Kayser cm-1
ws.f_grid = pyarts.arts.convert.kaycm2freq(f_grid_kayser)  # Convert to Hz

# The atmosphere consists of water, oxygen and nitrogen.
# We set these to be computed using predefined absorption
# models.
ws.abs_speciesSet(
    species=[
        "H2O, H2O-SelfContCKDMT400 , H2O- ForeignContCKDMT400",
        "O3",
        "CO2, CO2-CKDMT252",
        "CH4",
        "N2O",
    ]
)
ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")
ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)

# We now have all the information required to compute the absorption agenda.
ws.lbl_checkedCalc()  # Check that the line-by-line data is consistent
ws.propmat_clearsky_agendaAuto()

# Calculate lookup table
ws.abs_nls_interp_order = 3
ws.abs_t_interp_order = 3
ws.abs_p_interp_order = 3
h2o_perturbations = np.array([0.05, 0.25, 1, 4])
t_perturbations = np.linspace(-110, 50, num=4)
wsv.abs_lookupFascod(
    ws,
    fascod_atm=FascodVersion.TRO,
    f_grid=ws.f_grid.value.value,
    p_grid=ws.p_grid.value.value,
    t_perturbations=t_perturbations,
    h2o_nls=True,
    nls_perturbations=h2o_perturbations,
)

# We need an atmosphere.  This is taken from the arts-xml-data in
# raw format before being turned into the atmospheric fields that
# ARTS uses internally.
ws.AtmRawRead(basename="planets/Earth/Fascod/tropical/tropical")
ws.AtmFieldsCalc()

# These calculations do no partial derivatives, so we can turn it off
ws.jacobianOff()

# There is no scattering in this example, so we can turn it off
ws.cloudboxOff()

# The concept of a sensor does not apply to this example, so we can turn it off
ws.sensorOff()

# We check the atmospheric geometry, the atmospheric fields, the cloud box,
# and the sensor.
ws.atmgeom_checkedCalc()
ws.atmfields_checkedCalc()
ws.cloudbox_checkedCalc()
ws.sensor_checkedCalc()

# We perform the calculations
ws.timerStart()
ws.yCalc()
ws.timerStop()
time = ws.timer.value.cputime_end - ws.timer.value.cputime_start
print(time)

time_direct = 274819835
time_lookup = time
time_diff = time_direct - time_lookup
print(time_diff)

# Create a simple plot to look at the simulations.  Try increasing NF
# above to see more details
plt.plot(f_grid_kayser, ws.y.value.value.reshape(len(ws.sensor_pos.value.value), len(ws.f_grid.value.value)).T)
plt.xlabel("Kayser cm-1")
plt.ylabel("W/(m^2 Hz sr)")
plt.legend(["Looking down"])
plt.title("xx")
plt.show()
