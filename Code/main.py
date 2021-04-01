
#run this line:
#pyenv version
#if not python 3.7.9
#pyenv install 3.7.9
import numpy as np
import pandas as pd
from IPython.display import display     #for display

#from hello import *             #imports entire file
#from hello import my_function  #imports sepcific function



df = pd.read_excel("Data/1066528_heartrate.xlsx")
print(df)

