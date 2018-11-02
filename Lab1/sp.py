import csv
import math
from sklearn.model_selection import train_test_split

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def readFile(filename):
	dataset = []
	with open(filename,'r') as csvfile:
		readcsv = csv.reader(csvfile,delimiter=',')
		for row in readcsv:
			if not row:
				continue
			dataset.append(row)
	return dataset[1:]

def modifyDataset(dataset,n,typeA,typeB):
	for row in dataset:
		for column in range(0,n):
			row[column] = float(row[column].strip())
		if row[n] == typeA:
			row[n] = 0
		elif row[n] == typeB:
			row[n] = 1

def train(trainset,n,weights,lr,MAX_ITERATIONS,th):
	flag = 1
	count = 0
	while(flag and count<=MAX_ITERATIONS):
		count += 1
		flag = 0
		for row in trainset:
			pval = predict(row,n,weights,th)
			aval = row[n]
			error = aval - pval
			if error != 0:
				flag = 1
			weights[0] += lr*error
			for i in range(1,n+1):
				weights[i] += lr*error*row[i-1]

def findAccuracy(testset,n,weights,th):
	correct = 0
	for row in testset:
		pval = predict(row,n,weights,th)
		aval = row[n]
		if pval == aval:
			correct += 1
	return correct / float(len(testset)) * 100.0

def predict(row,n,weights,th):
	activation = weights[0]
	for i in range(n):
		activation += row[i] * weights[i+1]
	activation=sigmoid(activation)
	if activation >= th:
		return 1
	else:
		return 0

def main():
	#dataset = readFile('IRIS.csv')
	dataset = readFile('SPECT.csv')
	n = len(dataset[0])-1
	#modifyDataset(dataset,n,'Iris-setosa','Iris-versicolor')
	modifyDataset(dataset,n,'Yes','No')

	trainset,testset = train_test_split(dataset,test_size=0.33,random_state=42,shuffle=True)
	initialWgt = 1 / (n+1)
	weights = [initialWgt for i in range(n+1)]
	lrs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
	MAX_ITERATIONS = 1000
	accuracy={}
	ths = lrs
	for th in ths:
		for lr in lrs:
			train(trainset,n,weights,lr,MAX_ITERATIONS,th)
			per = findAccuracy(testset,n,weights,th)
			accuracy[per]=[lr,weights,th]
			weights = [initialWgt for i in range(n+1)]
	print(accuracy)
	maxAccuracy = max(accuracy)
	temp = accuracy[maxAccuracy]
	lr = temp[0]
	weights = temp[1]
	print("The max accuracy is",end=" : ")
	print(maxAccuracy,end="\n")
	print("The optimal learning rate is",end=" : ")
	print(lr,end="\n")
	print("The optimal weights are",end=" : ")
	print(weights,end="\n")




if __name__ == '__main__':
	main()