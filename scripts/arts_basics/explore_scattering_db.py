# %%
import os
import pyarts.xml as xml
import matplotlib.pyplot as plt

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

#%%
def get_attr(obj, attr_name, default=None):
    return [getattr(obj[i], attr_name, default) for i in range(len(obj))]
# %%
ext = get_attr(S, "ext_mat_data")
f_grid = get_attr(S, "f_grid")
# %%
get_attr(S, attributes[8])[0].shape
# plt.plot(d_veq, ext, marker=".")