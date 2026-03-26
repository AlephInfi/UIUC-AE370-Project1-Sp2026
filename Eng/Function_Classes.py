# import libraries
import numpy as np
import scipy
import scipy.linalg as spla

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

class body:
    __u = np.array([0,0,0])
    __v = np.array([0, 0, 0])
    __a = np.array([0, 0, 0])

    # assume vectors 0 if not specified or unknown
    def __init__(self,
                 u: np.ndarray | None = None,
                 v: np.ndarray | None = None,
                 a: np.ndarray | None = None):
        if u.shape != (3,) or v.shape != (3,) or a.shape != (3,):
            raise ValueError("u, v, and a must be 3-element vectors")

        # create fresh defaults if None
        u = np.zeros(3) if u is None else u
        v = np.zeros(3) if v is None else v
        a = np.zeros(3) if a is None else a

        self.__u = u
        self.__v = v
        self.__a = a

    def get_position(self):
        return self.__u
    
    def get_velocity(self):
        return self.__v
    
    def get_acceleration(self):
        return self.__a
    
    
    def leap_frog(u,v,a):
        for i in range(len(u-1)):
            v_halfstep = v[i] + (dt/2) * a[i]
            
            u[i+1] = u[i] + dt*v_halfstep

            a[i+1] = a(u[i+1])

            v[i+1] = v_halfstep + dt/2 * a[i+1]
        

    