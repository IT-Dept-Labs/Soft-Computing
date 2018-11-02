from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random

data="SPECT.csv"

df=pd.read_csv(data)

number=LabelEncoder()
df['class']=number.fit_transform(df['class'].astype('str'))


X=df.iloc[:,:df.shape[1]-1]

Y=df.iloc[:,df.shape[1]-1]





def train(X_train,Y_train):
	#print("Training with 80% of the data set:")
	size=X_train.shape[0]
	count1=0
	count0=0
	p1=[-1 for i in range(X_train.shape[1])]
	p2=[-1 for i in range(X_train.shape[1])]
	p3=[-1 for i in range(X_train.shape[1])]
	p4=[-1 for i in range(X_train.shape[1])]
	for i in range(size):
		if Y_train.iloc[i]==1:
			count1+=1
		else:
			count0+=1
	mp0=float(count0/size)
	mp1=float(count1/size)
	for j in range(X_train.shape[1]):
		v0y0=0
		v0y1=0
		v1y0=0
		v1y1=0
		c1=0
		c0=0
		for k in range(X_train.shape[0]):
			if Y_train.iloc[k]==0 and X_train.iloc[k][j]==0:
				v0y0+=1
			if Y_train.iloc[k]==1 and X_train.iloc[k][j]==0:
				v0y1+=1
			if Y_train.iloc[k]==0 and X_train.iloc[k][j]==1:
				v1y0+=1
			if Y_train.iloc[k]==1 and X_train.iloc[k][j]==1:
				v1y1+=1
			if X_train.iloc[k][j]==1:
				c1+=1
			if X_train.iloc[k][j]==0:
				c0+=1
		p1[j]=float(v0y0/count0)
		p2[j]=float(v0y1/count1)
		p3[j]=float(v1y0/count0)
		p4[j]=float(v1y1/count1)

	#print(p1,p2,p3,p4)
	return mp0,mp1,p1,p2,p3,p4

def testing(X_test,Y_test,mp0,mp1,p1,p2,p3,p4):
	#print("Testing:")
	truep=0
	falsep=0
	falsen=0

	count=0

	#print("p0 :{} p1 :{}".format(mp0,mp1))
	size=X_test.shape[0]
	for i in range(X_test.shape[0]):
		cr=X_test.iloc[i]
		y=Y_test.iloc[i]
		prod1=mp0
		prod2=mp1
		str0=""
		str1=""
		for j in range(len(cr)):
			if cr[j]==0:
				prod1=prod1*p1[j]
				prod2=prod2*p2[j]
				str0+=str(p1[j])+" "
				str1+=str(p2[j])+" "
			else:
				prod1=prod1*p3[j]
				prod2=prod2*p4[j]
				str0+=str(p3[j])+" "
				str1+=str(p4[j])+" "
	
		prediction=-1
		x1=prod1
		x2=prod2
		#print("x1:{} x2:{}".format(x1,x2))
		if(x1>x2):
			prediction=0
		else:
			prediction=1
		#print("Predicted value:{} whereas Actual value:{}".format(prediction,Y_test.iloc[i]))
		if(prediction==y and prediction==1):
			truep+=1
		if(prediction!=y and prediction==1):
			falsep+=1
		if(prediction!=y and prediction==0):
			falsen+=1
		if(prediction==Y_test.iloc[i]):
			count+=1
	
	return float(count/size),float(truep/(truep+falsep)),float(truep/(truep+falsen))

sumacc=0
sumpre=0
sumrec=0
for i in range(10):
	X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1000)
	mp0,mp1,p0,p1,p2,p3=train(X_train,Y_train)
	acc,pre,rec=testing(X_test,Y_test,mp0,mp1,p0,p1,p2,p3)
	sumacc+=acc
	sumpre+=pre
	sumrec+=rec
	

print("Accuracy is:{}".format(sumacc/10))
print("Precison is:",sumpre/10)
print("Recall is:",sumrec/10)

