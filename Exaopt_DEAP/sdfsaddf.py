'''
import numpy as np
import random

f = open('logbook2_solutions.log', 'r')
lines = f.readlines()[1:]
print(lines)
fn=np.array(lines)
print(fn)
f.close()

S_sim=np.array([1,2,124])#,3,45,62,1,23,5,32,54,2])
S_exp=np.array([1,32,2])#,23,2,13,2,1,3,45,32,5])

F1 = np.sqrt(sum((1 - (S_sim/S_exp))**2))
print(F1)
F2 = np.sqrt(sum((S_sim-S_exp)**2)/sum(S_exp**2))
print(F2)

print(sum((1 - (S_sim/S_exp))**2))
print(sum((S_sim-S_exp)**2)/sum(S_exp**2))
print(sum((S_sim-S_exp))**2/sum(S_exp)**2)
'''



import numpy
import pickle

NPOP = 52

# Read file
output="checkpoint_files/output_gen_50.pkl"

# Retrieve the state of the specified checkpoint
with open(output,"rb+") as ckp_file:
    ckp = pickle.load(ckp_file)

pop_fit = ckp["pop_fit"]

    