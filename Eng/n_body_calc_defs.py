import numpy as np
from Eng import Function_Classes

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

v = 1

#TODO: see if we need to add more here... we probably do