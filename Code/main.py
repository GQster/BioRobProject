#run this line:
#pyenv version
#if not python 3.7.9
#pyenv install 3.7.9
import numpy as np
import pandas as pd

from hello import *             #imports entire file
#from hello import my_function  #imports sepcific function

name = "bob"

my_function(name)

raw_array= [[1,2,3],[3,2,1]]
raw_array = np.array(raw_array)
print(np.mean(raw_array))
 