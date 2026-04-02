# %%
import os
import earthaccess

# %%
earthaccess.login()

# %%
results = earthaccess.search_data(
    short_name="GPM_MERGIR",
    temporal=("2025-08-01", "2026-03-01"),
)  # You can also give a more detailed period, e.g. ("2025-07-02", "2025-07-03")
earthaccess.download(results, "/scratch/li/cpcir")
