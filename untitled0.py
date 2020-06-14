import numpy as np
import pandas as pd

def entropy(c1, c2):
   return  -c1/(c1+c2)* np.log2(c1/(c1+c2)) - c2/(c1+c2)*np.log2(c2/(c1+c2))

def info_gain(g1, g2, c1, c2, c3, c4):
    return entropy(g1, g2) - (c1+c2)/(g1+g2) * entropy(c1, c2) - (c3+c4)/(g1+g2)* entropy(c3, c4)
    
data = pd.read_csv('ml-bugs.csv')
g1 = 10
g2 = 14

group = data.groupby('less20')['Species'].value_counts()
c1 = group[(True, 'Mobug')]
c2 = group[(True, 'Lobug')]
c3 = group[(False, 'Mobug')] 
c4 = group[(False, 'Lobug')] 
print(info_gain(g1,g2,c1,c2,c3,c4))