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