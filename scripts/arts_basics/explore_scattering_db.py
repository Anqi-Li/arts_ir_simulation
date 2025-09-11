# %%
import os
import pyarts.xml as xml

# %%
datafolder = "/data/s5/scattering_data/IRDbase/Yang2016/ArtsFormat"
habit = "5-PlateAggregate-Smooth"

# %%
M = xml.load(os.path.join(datafolder, f"{habit}.meta.xml"), search_arts_path=False)
attributes = [attr for attr in dir(M[0]) if not callable(getattr(M[0], attr)) and not attr.startswith("__")]
print(attributes)
print(len(M))

# %%
d_veq = [getattr(M[i], "diameter_volume_equ", None) for i in range(len(M))]
# %%
S = xml.load(os.path.join(datafolder, f"{habit}.xml"), search_arts_path=False)
attributes = [attr for attr in dir(S[0]) if not callable(getattr(S[0], attr)) and not attr.startswith("__")]
print(attributes)
print(len(S))

# %%
