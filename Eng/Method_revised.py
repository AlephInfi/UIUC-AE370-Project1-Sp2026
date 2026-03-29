from asyncio.windows_events import NULL

import numpy as np
from Eng.Functions import *

# BEGIN CONSTANTS
const_G = 6.67430 * 10**(-11) # m^3/(kg s^2)
const_m_Earth = 5.972 * 10**24 # kg
const_m_Moon = 7.348 * 10**22 # kg
const_m_sat = 1000 # kg
const_m_asteroid = 1 * 10**10 # kg
const_r_Moon = 3.844 * 10**8 # m
const_r_Earth = 6371 * 1000 # m
const_r_sat_init = const_r_Earth + (400 * 1000) # m

def find_v_sqrt_GM_r(M, r):
    return np.sqrt(const_G * M / r)

v_moon = find_v_sqrt_GM_r(const_m_Earth, const_r_Moon) # ~ 1.02 * 10^3
# END CONSTANTS

# store positional data and mass
class body:
    def __init__(self, u, v, a, m):
        self.__u = u
        self.__v = v
        self.__a = a
        self.__m = m

    def setData(self, u:list, v:list, a:list):
        if u != NULL:
            self.__u = u
        if v != NULL:
            self.__v = v
        if a != NULL:
            self.__a = a

    def getData(self):
        return [self.__u, self.__v, self.__a]

    def getMass(self):
        return self.__m

def gravity(m1, m2, dist_vector):
    rSq = dist_vector[0]**2 + dist_vector[1]**2 + dist_vector[2]**2
    return (const_G * m2 * m1) / rSq

def calculateGravAccel(target, bodies:list):
    """
    target - instance of body() class to update accelerations
    bodies - list of other bodies imposing grav force
    """
    targetData = target.getData()
    targetM = target.getMass()
    new_Accel = targetData[2] # start with prev. acceleration

    for i in bodies:
        posData = i.getData()
        u = np.array(posData[0])
        vector_between = np.subtract(u - targetData[0])         # vector from body to other body
        if np.isclose(np.linalg.norm(vector_between), 0):       # skips gravity calc if it's going to be 0
            continue
        force = gravity(targetM, i.getMass(), vector_between)   # execute F_G = Gmm/r2
        new_Accel += force / targetM                            # TODO: check if this is right?

# From Functions.py, combine the rk4 functions to make them denser and readable here
def rk4_combined():
    pass

# use rk4() and leapfrog(), find next position and return them both
def findNextPos():
    rk4_res, lf_res = [0, 0, 0]

    return rk4_res, lf_res

