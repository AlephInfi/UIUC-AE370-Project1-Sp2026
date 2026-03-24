# import libraries at the top
import numpy as np

# in this file you can define classes or whatever you need for your function to reference/instantiate
import Function_Classes

# obj is a function-local instance of the FuncHelperEx class
def function(num: int, obj: Function_Classes.FuncHelperEx):
    # add num and private variable from your function class
    return (float)(num + obj.getprivatevariable())