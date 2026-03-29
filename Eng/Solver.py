import numpy as np

from Eng.Functions import *
from Eng.n_body_calc_defs import *

class Solver:
    __bodies = {} # dict of bodies in the class 'body'
    __obj_sat = None
    __dt = 0

    def __init__(self, input, satellite_input, dt):
        for i in input:
            if len(i) != 6:
                raise ValueError(f"Check {i[0]}'s array size!")
            if len(i[3]) != 3 or len(i[4]) != 3 or len(i[5]) != 3:
                raise ValueError(f"{i[0]}'s u, v, and a must be 3-element vectors")
            if not np.isscalar(i[1]) or not np.isscalar(i[2]):
                raise ValueError(f"Mass and/or keep-out radius of {i[0]} must be a number!")

            self.__bodies[f"{i[0]}"] = body(i[1], i[2], i[3], i[4], i[5]) # init bodies with their own initial m, r, u, v, a

        if (len(satellite_input) != 7):
            raise ValueError("Check satellite input array size")
        if not np.isscalar(satellite_input[0]) or not np.isscalar(satellite_input[1]):
            raise ValueError("Satellite mass and keep-out radius must be a scalar")
        if len(satellite_input[2]) != 2:
            raise ValueError("Solver supports only longitude and latitude together!")
        if not np.isscalar(satellite_input[2][0]):
            raise ValueError("Longitude must be a number!")
        if not np.isscalar(satellite_input[2][1]):
            raise ValueError("Latitude must be a number!")
        for i in satellite_input[4:5]:
            if len(i) != 3:
                raise ValueError(f"Satellite initial v and a must be 3-element vectors")
            for ii in i:
                if not np.isscalar(ii):
                    raise ValueError(f"Satellite v and a components must be scalars!")
        if not np.isscalar(satellite_input[6]):
            raise ValueError("Satellite fuel mass flow rate must be a scalar!")
        if not np.isscalar(satellite_input[7]):
            raise ValueError("Satellite fuel I_sp must be a scalar!")

        self.__obj_sat = Satellite(satellite_input)

        self.__dt = dt

    # TODO: ensure this is okay and works? Need to finish too. implement rk4/leapfrog steps
    def Update(self):
        for dt in self.__dt:
            current_sat_dynam_data = self.obj_sat.getDynamData()
            curr_Moon_dynam_data = self.__bodies["Moon"].getDynamData()

            # Using current pos, vel, acc, get next frame's position
            sat_next = self.__obj_sat.updateDynamData(self.__bodies, dt)

        #