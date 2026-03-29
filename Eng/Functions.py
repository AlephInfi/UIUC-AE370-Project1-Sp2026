# import libraries at the top
import numpy as np

# in this file you can define classes or whatever you need for your function to reference/instantiate
from Eng.Function_Classes import *
from Eng.n_body_calc_defs import *

# obj is a function-local instance of the FuncHelperEx class
def function(num: int, obj: Function_Classes.FuncHelperEx):
    # add num and private variable from your function class
    return (float)(num + obj.getprivatevariable())

#########################################################################

# called in update(), use masses and current positions to alter acceleration from gravities of other bodies.
def compute_accelerations(masses, positions):
    # positions: shape (N, 3)
    # velocities: shape (N, 3)
    # masses: shape (N,1)   
    N = len(masses)
    accelerations = np.zeros((N, 3))
    r = np.zeros((N, N, 3))
    for i in range(N):
        ax, ay, az = 0.0, 0.0, 0.0

        for j in range(N):
            if i == j:
                continue

            dx = positions[j][0] - positions[i][0] # x-displacement
            dy = positions[j][1] - positions[i][1] # y-displacement
            dz = positions[j][2] - positions[i][2] # z-displacement

            r_sqaured = dx**2 + dy**2 + dz**2 + 1e-12 # I am adding this just to make sure that acceration doesnt go to infinity if the are too close
            r_cubic = r_sqaured * np.sqrt(r_sqaured)
            gravity_equation = const_G * masses[j] / r_cubic
            # compute acceration in each direction
            ax += gravity_equation * dx
            ay += gravity_equation * dy
            az += gravity_equation * dz

        accelerations[i] = [ax, ay, az]
    return accelerations

# to make thing easier for later RK4 implementation
def derivatives(positions, velocities, masses):
    accelerations = compute_accelerations(masses, positions)
    return velocities, accelerations

# RK4 equation yn+1 = yn + h/6(k1+2k2+2k3+k4) where ki = [[v], [a]]
def rk4_step(positions, velocities, masses, dt):

    # k1
    v1, a1 = derivatives(positions, velocities, masses)

    # k2
    v2, a2 = derivatives(
        positions + 0.5 * dt * v1,
        velocities + 0.5 * dt * a1,
        masses
    )

    # k3
    v3, a3 = derivatives(
        positions + 0.5 * dt * v2,
        velocities + 0.5 * dt * a2,
        masses
    )

    # k4
    v4, a4 = derivatives(
        positions + dt * v3,
        velocities + dt * a3,
        masses
    )

    # update
    positions_new = positions + dt/6 * (v1 + 2*v2 + 2*v3 + v4)
    velocities_new = velocities + dt/6 * (a1 + 2*a2 + 2*a3 + a4)

    return positions_new, velocities_new

# We will always start at t=0 and T is the final velocity
def rk4_final(positions, velocities, masses, dt, T):
    position = positions.copy()
    velocity = velocities.copy()
    traj = [position.copy()]
    traj_derv = [velocity.copy()]
    T_current = 0 # secs
    while (T_current < T):
        position, velocity = rk4_step(position, velocity, masses, dt)        
        traj.append(position.copy())
        traj_derv.append(velocity.copy())
        T_current += dt # maybe some float pointing drift

    return np.array(traj), np.array(traj_derv)


### LEAPFROG

def leapfrog_step(us, vs, masses, dt):
    a_s = compute_accelerations(masses, us)
    us_new = us + vs * dt + 0.5 * a_s * dt**2
    as_new = compute_accelerations(masses, us_new)
    vs_new = vs + 0.5 * (a_s + as_new) * dt

    return us_new, vs_new

### SYSTEM SIM

def simulate_system(bodies, dt_array, method="both"):
    body_list = list(bodies.values())
    N = len(body_list)

    # Extract initial conditions
    positions = np.array([b.getDynamData()[0] for b in body_list])
    velocities = np.array([b.getDynamData()[1] for b in body_list])
    masses = np.array([b._body__m for b in body_list])  # accessing private var

    # Storage
    traj_rk4, traj_lf = [], []
    vel_rk4, vel_lf = [], []

    pos_rk4 = positions.copy()
    vel_rk4_curr = velocities.copy()

    pos_lf = positions.copy()
    vel_lf_curr = velocities.copy()

    for dt in dt_array:

        if method in ["rk4", "both"]:
            pos_rk4, vel_rk4_curr = rk4_step(pos_rk4, vel_rk4_curr, masses, dt)
            traj_rk4.append(pos_rk4.copy())
            vel_rk4.append(vel_rk4_curr.copy())

        if method in ["lf", "both"]:
            pos_lf, vel_lf_curr = leapfrog_step(pos_lf, vel_lf_curr, masses, dt)
            traj_lf.append(pos_lf.copy())
            vel_lf.append(vel_lf_curr.copy())

    return {
        "rk4": (np.array(traj_rk4), np.array(vel_rk4)),
        "lf": (np.array(traj_lf), np.array(vel_lf))
    }