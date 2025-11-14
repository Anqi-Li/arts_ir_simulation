# %%
import math
import numpy as np
import matplotlib.pyplot as plt
from earthcare_ir import PSD
from psd import get_ws_psd
from physics.unitconv import deq2dgeo, dgeo2deq


# %%
def psd_F05_prior(dmeq, t, iwc):
    n0star = get_n0star_from_temperature(t=t)
    d0star = get_d0star_from_n0star_iwc(n0star=n0star, iwc=iwc)
    x = dmeq / d0star
    return n0star * psd_F05_norm(x)


def psd_F05_norm(x):
    # Following Field et al. (2005) Table 2 first row
    lambda_0 = 20.78
    lambda_1 = 3.29
    nu = 0.6357
    return 490.6 * np.exp(-lambda_0 * x) + 17.46 * (x**nu) * np.exp(-lambda_1 * x)


def get_n0star_from_temperature(t):
    # Following Mason et al 2023 Table 1
    t_c = t - 273.15
    a_v = np.exp(-6.9 + 0.0315 * t_c)
    n0prime = get_n0prime_from_temperature(t)
    return n0prime * a_v**0.6


def get_n0prime_from_temperature(t):
    t_c = t - 273.15
    return np.exp(16.118 - 0.1303 * t_c)


def get_d0star_from_n0star_iwc(iwc, n0star):
    # Compute d0star from n0star and iwc
    # Following Delanoë et al. (2014) Eq. (9)
    # Using the relation: iwc = (pi/6) * rho * n0star * d0star^4 * C
    # where C is a constant, arbitrarily chosen by Testud et al. (2001) to be equal to gamma(4)/4^4

    rho = 1000  # kg/m^3  # used in Delanoë et al. (2014)
    # C = math.gamma(4) / 4**4  # constant from Testud et al. (2001)
    C = 1
    d0star = (6 * iwc / (np.pi * rho * n0star * C)) ** (1 / 4)
    return d0star


# % Verify that integrating the PSD returns the correct IWC
rho = 1e3
iwc = 0.01 * 1e-3
t = 220  # K
d_meq = np.logspace(-6, -2, 10001)  # melted equivalent spherical diameter

psd_data = psd_F05_prior(d_meq, t=t, iwc=iwc)  # m^-4

# test consistency
iwc_from_psd_integral_dmeq = np.trapezoid(
    np.pi * rho / 6 * d_meq**3 * psd_data, d_meq
)  # kg/m^3
print(f"IWC from PSD integration (using dmeq): {iwc_from_psd_integral_dmeq:.2e} kg/m^3")

# test consistency using dmax
a, b = 480, 3
d_max, ddmeq_ddmax = deq2dgeo(d_meq, a=a, b=b, rho=rho)  # m
iwc_from_psd_integral_dmax = np.trapezoid(
    a * d_max**b * psd_data * ddmeq_ddmax, d_max
)  # kg/m^3
print(f"IWC from PSD integration (using dmax): {iwc_from_psd_integral_dmax:.2e} kg/m^3")

# show original IWC
print(f"IWC given: {iwc:.2e} kg/m^3")


# %%
def get_psd_moment_trapz(order, psd_data, size_grid, axis=-1):
    """Moment using trapezoidal rule.
    order: moment order
    psd_data: PSD data array
    size_grid: size grid array
    axis: axis along which to compute the moment
    """
    integrand = psd_data * size_grid**order
    return np.trapezoid(integrand, size_grid, axis=axis)


def get_n0star(i, j, psd_data, size_grid):
    # Compute the normalized number concentration
    # N0/N0star = normalised F(x)
    n0star = get_psd_moment_trapz(i, psd_data, size_grid) ** (
        (j + 1) / (j - i)
    ) * get_psd_moment_trapz(j, psd_data, size_grid) ** ((i + 1) / (i - j))
    return n0star


def get_d0star(i, j, psd_data, size_grid):
    # Compute the normalized median diameter
    # x = d/d0star
    d0star = (
        get_psd_moment_trapz(i, psd_data, size_grid)
        / get_psd_moment_trapz(j, psd_data, size_grid)
    ) ** (-1 / (j - i))
    return d0star


# %% Two plots side by side: normal and normalised PSD

a, b, rho = 0.21, 2.26, 1000  # values in plate aggregate from database
d_max = np.logspace(-6, -2, 100)  # m
d_meq, ddmeq_ddmax = dgeo2deq(d_max, a=a, b=b, rho=rho)  # m
t = 250  # K
iwc = 1e-4  # kg/m^3

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

for psd in [PSD.D14, PSD.F07T, PSD.MDG]:
    if psd == PSD.F07M or psd == PSD.F07T or psd == PSD.MDG:
        d = d_max
    else:
        d = d_meq

    _, psd_data = get_ws_psd(
        fwc=np.atleast_1d(iwc),  # kg/m^3
        t=np.atleast_1d(t),  # K
        psd=psd,
        mgd_coef={"n0": 1e10, "ga": 1.5, "mu": 0},
        scat_species_a=a,
        scat_species_b=b,
        psd_size_grid=d,
        rho=rho,
    )
    psd_data = psd_data.value.squeeze()  # m^-4

    # make sure all psd are in terms of d_meq
    if psd == PSD.F07T or psd == PSD.F07M or psd == PSD.MDG:
        psd_data *= ddmeq_ddmax

    # Left plot: Normal PSD
    ax1.loglog(
        d_meq, psd_data.squeeze(), marker=".", markersize=1, linewidth=0.5, label=psd
    )

    # Right plot: Normalized PSD
    n0star = get_n0star(i=2, j=3, psd_data=psd_data, size_grid=d_meq)
    d0star = get_d0star(i=2, j=3, psd_data=psd_data, size_grid=d_meq)
    psd_data_norm = psd_data / n0star
    x = d_meq / d0star
    ax2.semilogy(
        x, psd_data_norm.squeeze(), marker=".", markersize=1, linewidth=0.5, label=psd
    )
    print(psd)
    iwc_from_psd_integral = np.trapezoid(np.pi * rho / 6 * d_meq**3 * psd_data, d_meq)
    print(f"IWC from psd integration over dmeq: {iwc_from_psd_integral:.3e} kg/m^3")
    print("ratio to given IWC:", iwc_from_psd_integral / iwc)
    iwc_from_psd_integral = np.trapezoid(a * d_max**b * psd_data / ddmeq_ddmax, d_max)
    print(f"IWC from psd integration over dmax: {iwc_from_psd_integral:.3e} kg/m^3")
    print("ratio to given IWC:", iwc_from_psd_integral / iwc)

    ax3.plot(
        d_meq,
        psd_data.squeeze() * d_meq**6,
        marker=".",
        markersize=1,
        linewidth=0.5,
        label=psd,
    )

# ACMCAP for both plots
psd_data = psd_F05_prior(d_meq, t=t, iwc=iwc)  # m^-4
ax1.loglog(d_meq, psd_data, marker=".", markersize=1, linewidth=0.5, label="ACM_CAP")

x = d_meq / get_d0star_from_n0star_iwc(n0star=get_n0star_from_temperature(t=t), iwc=iwc)
ax2.semilogy(
    x, psd_F05_norm(x), marker=".", markersize=1, linewidth=0.5, label="ACM_CAP"
)
print("ACM_CAP")
iwc_from_psd_integral = np.trapezoid(np.pi * rho / 6 * d_meq**3 * psd_data, d_meq)
print(f"IWC from psd integration over dmeq: {iwc_from_psd_integral:.3e} kg/m^3")
print("ratio to given IWC:", iwc_from_psd_integral / iwc)
iwc_from_psd_integral = np.trapezoid(a * d_max**b * psd_data / ddmeq_ddmax, d_max)
print(f"IWC from psd integration over dmax: {iwc_from_psd_integral:.3e} kg/m^3")
print("ratio to given IWC:", iwc_from_psd_integral / iwc)

ax3.plot(
    d_meq,
    psd_data.squeeze() * d_meq**6,
    marker=".",
    markersize=1,
    linewidth=0.5,
    label="ACM_CAP",
)

# Figure formatting
fig.suptitle(f"IWC={iwc:.2e} kg/m^3, T={t:.1f} K \na={a:.3f}, b={b:.3f}")

# Left plot formatting
ax1.set_xlabel("$D_{meq}$ (m)")
ax1.set_ylabel("$N(D)$ (m$^{-4}$)")
ax1.set_title(f"PSD")
ax1.set_ylim(1e-0, 1e12)
ax1.legend()
ax1.grid()

# Right plot formatting
ax2.set_xlabel("$x = D_{meq}/D_0^*$")
ax2.set_ylabel("$N(D_{meq})/N_0^*$")
ax2.set_title("Normalized PSD")
ax2.set_ylim(1e-5, 1e3)
ax2.set_xlim(0, 5)
ax2.legend()
ax2.grid()

ax3.set_xlabel("$D_{meq}$ (m)")
ax3.set_ylabel("$N(D) \cdot D_{meq}^6$ (m$^{2}$)")
ax3.set_title(f"6th Moment of PSD")
ax3.legend()
ax3.grid()

plt.tight_layout()
plt.show()


# %% test 3rd and 4th moment of normalised psd equal to IWC

# %% test changing n0star effects on ACM_CAP PSD
t_c = np.linspace(0, -80, 100)
t = t_c + 273.15

# plot n0star and n0prime vs temperature
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(get_n0star_from_temperature(t), t_c)
ax[0].set_xscale("log")
ax[0].set_xlabel("N0star (m^-4)")
ax[0].set_ylabel("Temperature (°C)")
ax[0].invert_yaxis()
ax[0].grid()

ax[1].plot(get_n0prime_from_temperature(t), t_c)
ax[1].set_xscale("log")
ax[1].set_xlabel("N0prime (m^-4)")
ax[1].set_ylabel("Temperature (°C)")
ax[1].invert_yaxis()
ax[1].grid()
plt.show()

# %% plot ACM_CAP PSD for different temperatures at fixed IWC
# to see how n0star and d0star change with temperature
# and how that affects the PSD shape
# fix IWC at 1e-4 kg/m^3
t_c = [0, -20, -40, -60]  # °C
t = np.array(t_c) + 273.15
iwc = 1e-4  # kg/m^3
n0star_values = [get_n0star_from_temperature(ti) for ti in t]
d_meq = np.logspace(-6, -2, 100)  # m
colors = colors = ["#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
ls = ["-", "--", "-.", ":"]
for i, n0star in enumerate(n0star_values):
    d0star = get_d0star_from_n0star_iwc(n0star=n0star, iwc=iwc)
    x = d_meq / d0star
    psd_cap = n0star * psd_F05_norm(x)
    plt.loglog(
        d_meq,
        psd_cap,
        c=colors[i],
        ls=ls[i],
        lw=5,
        label=f"$T$={t_c[i]} °C, $N_0*$={n0star:.2e}, $D_0*$={d0star*1e3:.2f} mm",
    )

plt.ylim(1e0, 1e12)
plt.xlabel(r"$D_{meq}$ (m)")
plt.ylabel(r"$N(D_{meq})$ (m$^{-4}$)")
plt.title(f"ACM_CAP PSD for different temperatures \nat IWC={iwc:.2e} kg/m^3")
plt.legend()
plt.grid()
# %%
