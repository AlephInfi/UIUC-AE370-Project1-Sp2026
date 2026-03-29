# import libraries
import numpy as np
import scipy
import scipy.linalg as spla

from Eng.n_body_calc_defs import *

# you can define classes with their own variables if you're
# instantiating a lot of unique things in a function, i.e.
# custom vector math, vertices in 3d space, etc.
class FuncHelperEx:
    __private_variable = 0
    public_variable = 1

    def __init__(self, num):
        self.__private_variable = num

    def getprivatevariable(self):
        return self.__private_variable

# WRITE CUSTOM CLASSES BELOW #

# TODO: How do we link the free bodies and satellite together? All bodies are assumed to affect each other.
# Idea: we should calculate the influence of gravity from each object at every frame and use it to update.

class body:
    __u = np.array([0,0,0])
    __v = np.array([0, 0, 0])
    __a = np.array([0, 0, 0])
    __m = 0

    # assume vectors 0 if not specified or unknown
    def __init__(self,
                 m,
                 r,
                 u: np.ndarray | None = None,
                 v: np.ndarray | None = None,
                 a: np.ndarray | None = None):
        if u.shape != (3,) or v.shape != (3,) or a.shape != (3,):
            raise ValueError("u, v, and a must be 3-element vectors")

        # create fresh defaults if None
        u = np.zeros(3) if u is None else u
        v = np.zeros(3) if v is None else v
        a = np.zeros(3) if a is None else a

        if not np.isscalar(m):
            raise ValueError("m must be a scalar")

        self.__m = m
        self.__u = np.array(u)
        self.__v = np.array(v)
        self.__a = np.array(a)

    # return current dynamics data
    def getDynamData(self):
        return [self.__u, self.__v, self.__a]

    # TODO: input a list of body data from other objects, calculate changes in pos, vel, accel
    def updateDynamData(self, other_bodies: list, dt: float):
        pass

class Satellite:
    # when the satellite is released, a burn must be conducted, changing its position.
    # when burnout is concluded, the natural trajectory of the satellite should allow
    # it to meet up with the asteroid and effectively collide
    __m_sat = 0
    __r_sat = 0
    __u_sat = [0, 0, 0]
    __v_sat = [0, 0, 0]
    __a_sat = [0, 0, 0]
    __mdot_fuel_sat = 0
    __I_sp_sat = 0

    def __init__(self, satellite_input: list):
        self.__m_sat = satellite_input[0]
        self.__r_sat = satellite_input[1]

        # calculate satellite initial position using longitude, latitude, and height data contained in sat_input[2]
        lat_deg, lon_deg, alt = satellite_input[2]
        # convert to radians
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)
        # radius from Earth's center
        r = const_r_Earth + alt
        # compute coordinates
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.sin(lat)
        z = r * np.cos(lat) * np.sin(lon)

        self.__u_sat = np.array([x, y, z])
        self.__v_sat = np.array(satellite_input[3])
        self.__a_sat = np.array(satellite_input[4])
        self.__mdot_fuel_sat = satellite_input[5]
        self.__I_sp_sat = satellite_input[6]

    # return current dynamics data
    def getDynamData(self):
        return [self.__u_sat, self.__v_sat, self.__a_sat]

    # TODO: should account for mass flow rate and also calculate trajectory based on acceleration, which is
    # affected by the changing mass.
    def updateDynamData(self, other_bodies: list, dt: float):
        pass

    def leap_frog(u, v, a, dt):
        for i in range(len(u - 1)):
            v_halfstep = v[i] + (dt / 2) * a[i]

            u[i + 1] = u[i] + dt * v_halfstep

            a[i + 1] = a(u[i + 1])

            v[i + 1] = v_halfstep + dt / 2 * a[i + 1]

