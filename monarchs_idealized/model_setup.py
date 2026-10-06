"""
Idealized MONARCHS run: a gently sloping ice-shelf surface with a single
closed depression downslope.

Meltwater produced across the domain should percolate, saturate the firn,
then flow laterally downslope and pond in the depression - the melt ->
drainage -> redistribution process this repo is interested in.

Run with:
    monarchs -i model_setup.py
or:
    python model_setup.py
"""

import numpy as np

"""
Spatial parameters
"""
row_amount = 15  # rows (y), looking top-down
col_amount = 15  # columns (x)
lat_grid_size = 1000  # lateral grid cell size [m]
vertical_points_firn = 200
vertical_points_lake = 20
vertical_points_lid = 20

"""
Timestepping
"""
warm_days = 50
cold_days = 50
num_days = warm_days + cold_days
t_steps_per_day = 24
lateral_timestep = 3600 * t_steps_per_day

"""
Idealized surface. MONARCHS uses firn_depth as surface elevation.
Plane sloping down toward +x (5 m over the domain), plus a Gaussian
basin 4 m deep centred two-thirds of the way downslope.
"""
x = np.arange(col_amount) * lat_grid_size
y = np.arange(row_amount) * lat_grid_size
X, Y = np.meshgrid(x, y)
base_height = 45.0
slope_drop = 5.0
basin_depth = 4.0
basin_sigma = 2000.0  # [m]
basin_x0, basin_y0 = x[-1] * 2 / 3, y[-1] / 2

firn_depth = (
    base_height
    - slope_drop * X / x[-1]
    - basin_depth
    * np.exp(-((X - basin_x0) ** 2 + (Y - basin_y0) ** 2) / (2 * basin_sigma**2))
)

rho_init = "default"
T_init = "default"
rho_sfc = 500
firn_max_height = 100
firn_min_height = 30
max_height_handler = "filter"
min_height_handler = "extend"

"""
Met forcing: spatially uniform, constant warm period then cold period.
"""
nt = num_days * t_steps_per_day
warm = warm_days * t_steps_per_day
cold = cold_days * t_steps_per_day


def warm_cold(warm_val, cold_val):
    return np.concatenate([warm_val * np.ones(warm), cold_val * np.ones(cold)])


met_data = {
    "LW_down": warm_cold(800, 100),  # [W m^-2]
    "SW_down": warm_cold(800, 100),  # [W m^-2]
    "temperature": warm_cold(267, 250),  # [K]
    "dew_point_temperature": warm_cold(265, 240),  # [K]
    "surf_pressure": 1000 * np.ones(nt),  # [hPa]
    "wind": 5 * np.ones(nt),  # [m s^-1]
    "snowfall": np.zeros(nt),  # [m s^-1]
    "snow_dens": 300 * np.ones(nt),  # [kg m^-3]
}
for key in met_data:
    met_data[key] = np.broadcast_to(
        met_data[key][:, np.newaxis, np.newaxis], (nt, row_amount, col_amount)
    )

"""
Output
"""
met_output_filepath = "output/met_data_idealized.nc"
save_output = True
vars_to_save = (
    "firn_depth",
    "lake_depth",
    "lid_depth",
    "lake",
    "lid",
    "v_lid",
    "ice_lens_depth",
    "water_level",
    "water_direction",
    "Lfrac",
)
output_filepath = "output/idealized_output.nc"
output_grid_size = 20
output_timestep = 1  # days
dump_data = True
dump_filepath = "output/idealized_dump.nc"
reload_from_dump = False

"""
Numerics
"""
use_numba = True
parallel = True
cores = "all"
spinup = False
flow_speed_scaling = 1.0

"""
Toggles
"""
catchment_outflow = False  # keep water inside the domain
flow_into_land = True
lateral_movement_toggle = True
lake_development_toggle = True
lid_development_toggle = True
single_column_toggle = True

if __name__ == "__main__":
    from monarchs.core.driver import monarchs

    monarchs()
