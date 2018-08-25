import math
import numpy as np
from physics_sim import PhysicsSim

def rotate_by_unit_vector(x, y, x_hat, y_hat):
    # keep in mind: magntidue of x_hat, y_hat is 1.0
    proj_xy_on_hat = x*x_hat + y*y_hat
    # square_magnitude_of_xy = x*x + y*y
    # proj_xy_on_side = math.sqrt(square_magnitude_of_xy - proj_xy_on_hat*proj_xy_on_hat)
    # using abs() to overcome numerical errors near zero...
    proj_xy_on_side = math.sqrt(abs(x*x + y*y - proj_xy_on_hat*proj_xy_on_hat))
    # using cross product to calculate the sign of the sin of the angle between xy and hat
    if x_hat*y - y_hat*x < 0:
        proj_xy_on_side *= -1
    return proj_xy_on_hat, proj_xy_on_side

def is_sim_crash(sim):
    sim_pos = sim.pose[:3]
    return any(sim_pos <= sim.lower_bounds) or any(sim_pos >= sim.upper_bounds)

def is_sim_kissing_the_ground(sim, prev_speed):
    sim_pos = sim.pose[:3]
    if (sim_pos[2] > sim.lower_bounds[2]) or any(sim_pos[:2] <= sim.lower_bounds[:2]) or any(sim_pos >= sim.upper_bounds):
        return False
    # touching or crashing on the ground
    if prev_speed < 0.01:
        return True
    vx, vy, vz = sim.v[0], sim.v[1], sim.v[2]
    if math.sqrt(vx*vx+vy*vy+vz*vz) > 0.01: # 1 centimeter per second
        # flying too fast
        return False
    if any(abs(sim.pose[4:]) > 0.1):
        # too tilted
        return False
    vphi, vtheta, vpsi = sim.angular_v[0], sim.angular_v[1], sim.angular_v[2]
    if math.sqrt(vphi*vphi+vtheta*vtheta+vpsi*vpsi) > 1.: # radian per second
        # spiraling too fast
        return False
    return True

def is_sim_crash(sim):
    sim_pos = sim.pose[:3]
    return any(sim_pos <= sim.lower_bounds) or any(sim_pos >= sim.upper_bounds)

def is_sim_finite(sim):
    return np.isfinite(sim.pose).all() and np.isfinite(sim.v).all() and np.isfinite(sim.angular_v).all()


def normalize_by_characteristics(value, character):
    value = np.asarray(value) / character
    return value / (1+abs(value))