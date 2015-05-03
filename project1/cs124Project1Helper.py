import numpy as np
import scipy.stats as stat 
from copy import copy, deepcopy

dataIn = np.random.random_integers(0,1,(1000,200)) ## make 1000 rows, 200 columns, entries 0/1 

disease_status = np.random.binomial(1,0.1,1000)
disease_status.sort() # first 900 people has 0, so they don't have disease 

hasMinor = np.random.binomial(1,.25,900) # has minor allele in the controls 
hasMinor = np.append ( hasMinor, np.random.binomial(1,0.95,100) ) # has minor allele in the cases 
dataIn[:,0] = hasMinor # assume first SNP is causal, so the 0/1 is not randomly distributed. 

# Verify that the control and case have appropriate s1 
sum ( dataIn [0:899,0] )/900. # the dot. is needed when int divides int, to cast into decimal 
sum ( dataIn [900:999,0] )/100. 

np.corrcoef(dataIn[:,1],dataIn[:,2]) # correlation of snps s2 and s3 (remember, in python, indexing starts at zero.)
corMatrix = np.corrcoef( dataIn.transpose() ) ## correlation of the 200 Snps in the data. 
np.shape(corMatrix) # should be dim of 200x200

(np.sum(np.abs(corMatrix)>0.1) - 200) / 2 # number of pairs that have absolute correlation over 0.1 

index = np.where(np.abs(corMatrix)>0.1) # row/column index of where the abs cor is over 0.1     

def breed (dataIn,i,j): 
    ''' i mates j, make person k '''
    personI = dataIn[i,:]
    personJ = dataIn[j,:]
    ''' cross over happens '''
    flip = personI == personJ # same allele will be passed on. 
    flip = np.where(flip==False) # any position with different allele, the allele will be chosen randomly 
    # print ( np.shape(flip)[1] ) # how many times you need to do the flips 
    personK = deepcopy(personI) ## WILL NOT WORK . if you do personK = personI without deepcopy 
#!! this is a bit incorrect, the true process of 'cross-over' is not simple fliping of bits, usually, "chunks" are cross-over 
#!! you can modify this process. 
    ''' much more complicated if alleles are not independent '''
    personK[flip] = np.random.binomial( 1,.5,np.shape(flip)[1] )
    return personK
    
offspring1 = breed(dataIn,1,2)    # first offspring of pair 1,2
np.corrcoef(dataIn[1,:],offspring1)
np.corrcoef(dataIn[2,:],offspring1)

offspring2 = breed(dataIn,1,2)    # second offspring of pair 1,2
np.corrcoef(dataIn[1,:],offspring2)
np.corrcoef(dataIn[2,:],offspring2)

np.corrcoef(offspring1,offspring2) # correlation between sibs. 


