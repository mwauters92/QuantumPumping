from qutip import Qobj, tensor
import numpy as np

def mycreate(Mcut):
    
    op = np.zeros([Mcut,Mcut],dtype=complex)
    j = np.arange(Mcut-1)
    op[j+1,j] = 1.0
    
    return Qobj(op)

def mydestroy(Mcut):
    
    op = np.zeros([Mcut,Mcut],dtype=complex)
    j = np.arange(Mcut-1)
    op[j,j+1] = 1.0
    
    return Qobj(op)
