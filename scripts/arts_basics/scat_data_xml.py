#%%
import easy_arts.wsv as wsv
import easy_arts.easy_arts as ea
from easy_arts.data_model import (
    CloudboxOption,
    SpectralRegion,
)
import pyarts.arts as pa
import os

ws = ea.wsInit(SpectralRegion.TIR)

ws.stdhabits_folder = "/scratch/li/arts-yang-liquid-simlink"
habit = "MieSpheres_H2O_liquid_"

path = os.path.join(str(ws.stdhabits_folder.value), f"{habit}.xml")
scat = pa.ArrayOfSingleScatteringData()
scat.readxml(path)
#%%
ws.output_file_formatSetBinary()
ws.WriteXML(
    input=scat[0],
    filename=os.path.join(str(ws.stdhabits_folder.value), f"{habit}_.xml"),
)

#%%
path = os.path.join(str(ws.stdhabits_folder.value), f"{habit}.meta.xml")
meta = pa.ArrayOfScatteringMetaData()
meta.readxml(path)
#%%
ws.output_file_formatSetBinary()
ws.WriteXML(
    input=meta[0],
    filename=os.path.join(str(ws.stdhabits_folder.value), f"{habit}_.meta.xml"),
)