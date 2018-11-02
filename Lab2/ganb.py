import pandas as pd
from random import randint
import random
from p1 import train,testing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
random.seed(10)

df=pd.read_csv('SPECT.csv')
number=LabelEncoder()
df['class']=number.fit_transform(df['class'].astype('str'))
X=df.iloc[:,:df.shape[1]-1]
Y=df.iloc[:,df.shape[1]-1]



def create_population(number,numberc):
	population=[]
	for i in range(numberc):
		strx=""
		for j in range(number):
			strx+=str(randint(0,1))
		population.append(strx)
	return population

def fitness_evaluation(population):
	scores=[]
	prec=[]
	rec=[]
	for i in population:
		df1=X
		for j in range(len(i)):
			if i[j]=='0':
				df1=df1.drop(X.columns[j], axis=1)
		X_train,X_test,Y_train,Y_test=train_test_split(df1,Y,test_size=0.2,random_state=1000)
		mp0,mp1,p0,p1,p2,p3=train(X_train,Y_train)
		score,precision,recall=testing(X_test,Y_test,mp0,mp1,p0,p1,p2,p3)
		scores.append(score)
		prec.append(precision)
		rec.append(recall)
	return (scores,prec,rec)

def selection(f,population):
	probability_f=[]   #probability fitness
	selected_chromosome=[]
	random_v=[]
	for i in range(len(f)):
		probability_f.append(0)
	for i in range(len(f)):
		probability_f[i]=f[i]/(sum(f))
	cumulative_f=[]
	for i in range(len(f)):
		cumulative_f.append(0)
		random_v.append(0)
		selected_chromosome.append(0)
	for i in range(len(f)):
		for j in range(i+1):
			cumulative_f[i]=cumulative_f[i]+probability_f[j]
	for i in range(len(f)):
		random_v[i]=random.uniform(0,1)	
	for i in range(len(f)):
		k=0
		for j in range(len(f)):
			if(random_v[i]>cumulative_f[j]):
				k=k+1
		selected_chromosome[i]=k
	selected_chromosomes=[]
	for i in selected_chromosome:
		selected_chromosomes.append(str(population[i]))
	return selected_chromosomes




def crossover(population,rate):
	lucky=random.sample(range(0,len(population)),int(rate*len(population)))
	new_population=[]
	count=0
	for i in range(len(lucky)):
		for j in range(i+1,len(lucky)):
			point=random.sample(range(1,21),1)
			point=point[0]
			x=population[lucky[i]][:point]+population[lucky[j]][point:]
			y=population[lucky[j]][:point]+population[lucky[i]][point:]
			new_population.append(x)
			new_population.append(y)
			count+=2
			if(count>=30):
				#print("Lucky is:{} and Count is:{}".format(lucky,count))
				return new_population
	#print("Lucky is:{} and Count is:{}".format(lucky,count))
	return new_population

def mutation(new_population,rate):
	for i in range(len(new_population)):
		lucky=random.sample(range(0,len(new_population[0])),int(rate*len(new_population[0])))
		for j in lucky:
			if new_population[i][j]=='0':
				new = list(new_population[i])
				new[j] = '1'
				new_population[i]=''.join(new)
			else:
				new = list(new_population[i])
				new[j] = '0'
				new_population[i]=''.join(new)
	return new_population


popu=create_population(X.shape[1],30)
print("The population is:{}".format(popu))
for i in range(5):
	scores,precs,recs=fitness_evaluation(popu)
	print("The average accuracy of the population:{}".format(sum(scores)/len(scores)))
	# print("Average precision:",sum(precs)/len(precs))
	# print("Average recall:",sum(recs)/len(recs))
	print(" ")
	newpop=selection(scores,popu)
	new_popu=crossover(newpop,0.25)
	#print("The new population is:{}".format(new_popu))
	popu=mutation(new_popu,0.1)
	print("The new mutated population is:{}".format(popu))









