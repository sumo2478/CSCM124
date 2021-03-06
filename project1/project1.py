import numpy as np
import scipy.stats as stats
from copy import copy, deepcopy
import math

def calculateNCP(p_plus, p_minus, p, totalSize):
	return (p_plus - p_minus) / (math.sqrt(2/float(totalSize)) * math.sqrt(p*(1-p)))

def calculateSignificanceForSNP(snpData, disease_status):	
	# Calculate the number of case and controls
	totalSize = len(disease_status)
	caseTotal = np.count_nonzero(disease_status)
	controlTotal = totalSize - caseTotal

	numberOfCaseWithMinorAllele = 0
	numberOfControlWithMinorAllele = 0

	# For each data item in the SNP
	for idx, data in enumerate(snpData):
		# If the person has the disease add value to the case
		if disease_status[idx] == 1:
			numberOfCaseWithMinorAllele += data if data == 1 else 0
		# Otherwise add it to value to the control
		else:
			numberOfControlWithMinorAllele += data if data == 1 else 0

	# Calculate observed case
	p_plus = numberOfCaseWithMinorAllele / float(caseTotal)

	# Calculate observed control
	p_minus = numberOfCaseWithMinorAllele / float(controlTotal)

	# Calculate p
	p = (p_plus + p_minus) / 2

	significance = calculateNCP(p_plus, p_minus, p, totalSize)
	
	return significance

def calculateSignificance(snpData, disease_status):
	if len(snpData) != len(disease_status):
		print("Error: lengths don't match")
		return

	# Calculate the number of people with the disease
	maxSNPSignificance = -999

	numberOfSNPs = len(snpData[0])
	
	# For each SNP	
	for i in range(numberOfSNPs):
		snp = snpData[:, i]

		# Calculate the significance of that SNP
		significance = calculateSignificanceForSNP(snp, disease_status)

		# If it is the most significant SNP then set it to the max significance
		maxSNPSignificance = significance if significance > maxSNPSignificance else maxSNPSignificance 
	
	return maxSNPSignificance

def calculateThreshold(alpha, numberOfSNPs):
	correctedAlpha = alpha / float(numberOfSNPs)
	return stats.norm.ppf(correctedAlpha)

def problem1c(dataIn, disease_status, numberOfSNPs, alpha):
	print("Problem 1C ::")
	significance = calculateSignificance(dataIn, disease_status)
	threshold = calculateThreshold(alpha, numberOfSNPs)
	print("Significance: " + str(significance))
	print("Threshold: " + str(threshold) + "\n\n")

def findGreedyTagSolution(snpAssociations):
	numberOfAssoc = [0]*len(snpAssociations)
	tags = []

	# Calculate the number of related associations
	for idx, assoc in enumerate(snpAssociations):
		if assoc is not None:
			numberOfAssoc[idx] = len(assoc)

	# While there is a max association greater than 0
	maxAssoc = max(numberOfAssoc)	
	while (maxAssoc > 0):				
		indexOfMax = -1

		# Find the index of the max assoc
		for index, num in enumerate(numberOfAssoc):
			if num == maxAssoc:
				tags.append(index)
				indexOfMax = index
				break
		
		# Retrieve associations corresponding to index
		assoc = snpAssociations[indexOfMax]		
		
		for tag in assoc:
			numberOfAssoc[tag] = 0

		maxAssoc = max(numberOfAssoc)

	return tags


def problem1d(dataIn):
	print("Problem 1d :: ")
	# Determine number of SNPs
	numberOfSNPs = len(dataIn[0, :])

	# Determine number of pairs
	corMatrix = np.corrcoef( dataIn.transpose() ) ## correlation of the 200 Snps in the data. 	
	numberOfPairs = (np.sum(np.abs(corMatrix)>0.1) - 200) / 2 # number of pairs that have absolute correlation over 0.1 
	print("Number of pairs: " + str(numberOfPairs))

	# Determine the indices where the abs cor is over 0.1
	index = np.where(np.abs(corMatrix)>0.1) # row/column index of where the abs cor is over 0.1     
	xSNPs = index[0]
	ySNPs = index[1]

	snpAssociations = [None]*numberOfSNPs

	for i in range(len(xSNPs)):		
		xValue = xSNPs[i]
		yValue = ySNPs[i]

		assocContainer = None
		currentContainer = snpAssociations[xValue]

		if currentContainer is None:
			assocContainer = []
		else:
			assocContainer = snpAssociations[xValue]

		assocContainer.append(yValue)
		snpAssociations[xValue] = assocContainer		

	greedyTagSolution = findGreedyTagSolution(snpAssociations)
	
	print("Tag selection: " + str(greedyTagSolution) + "\n\n")

def calculatePower(alpha, numberOfSNPs, ncp):
	correctedAlpha = alpha / float(numberOfSNPs)

	return stats.norm.cdf(stats.norm.ppf(correctedAlpha/2) + ncp) + 1 - stats.norm.cdf(-1 * stats.norm.ppf(correctedAlpha/2) + ncp)	

def problem1e(dataIn, alpha, numberOfSNPs, disease_status):
	print("Problem 1e ::")

	# Calculate the NCP of s0
	ncpS0 = calculateSignificanceForSNP(dataIn[:, 0], disease_status)

	totalPower = 0
	# For each SNP
	for index in range(numberOfSNPs):
		# Find the correlation between tag s0 and the current tag
		correlation = np.corrcoef(dataIn[:, 0], dataIn[:, index])
		correlation = correlation[0][1]

		# Compute the NCP of of the current tag
		ncp = ncpS0 * correlation

		# Calculate the power for this SNP
		currentPower = calculatePower(alpha, numberOfSNPs, ncp)

		# Added it to the total power
		totalPower += currentPower

	# Return the average of all the powers
	averagePower = totalPower / float(numberOfSNPs)
	print("Power: " + str(averagePower) + "\n\n")
	
def problem1():
	print("=======================================\nProblem 1\n=======================================")
	numberOfSNPs = 200
	alpha = 0.05

	dataIn = np.random.random_integers(0,1,(1000,numberOfSNPs)) ## make 1000 rows, 200 columns, entries 0/1 

	disease_status = np.random.binomial(2,0.1,1000)
	disease_status.sort() # first 900 people has 0, so they don't have disease 

	hasMinor = np.random.binomial(1,.25,900) # has minor allele in the controls 
	hasMinor = np.append ( hasMinor, np.random.binomial(1,0.95,100) ) # has minor allele in the cases 
	dataIn[:,0] = hasMinor # assume first SNP is causal, so the 0/1 is not randomly distributed. 

	# Verify that the control and case have appropriate s1 
	sum ( dataIn [0:899,0] )/900. # the dot. is needed when int divides int, to cast into decimal 
	sum ( dataIn [900:999,0] )/100.	

	problem1c(dataIn, disease_status, numberOfSNPs, alpha)
	problem1d(dataIn)
	problem1e(dataIn, alpha, numberOfSNPs, disease_status)

def main():
	problem1()


	

if __name__ == '__main__':
	main()