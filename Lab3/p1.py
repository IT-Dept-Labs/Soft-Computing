

def train(X_train,Y_train):
	size=X_train.shape[0]
	count1=0
	count0=0
	col=X_train.shape[1]
	p1=[-1 for i in range(col)]
	p2=[-1 for i in range(col)]
	p3=[-1 for i in range(col)]
	p4=[-1 for i in range(col)]
	for i in range(size):
		if Y_train.iloc[i]==1:
			count1+=1
		else:
			count0+=1
	mp0=count0/size
	mp1=count1/size
	for j in range(col):
		v0y0=0
		v0y1=0
		v1y0=0
		v1y1=0
		for k in range(size):
			if Y_train.iloc[k]==0 and X_train.iloc[k][j]==0:
				v0y0+=1
			if Y_train.iloc[k]==1 and X_train.iloc[k][j]==0:
				v0y1+=1
			if Y_train.iloc[k]==0 and X_train.iloc[k][j]==1:
				v1y0+=1
			if Y_train.iloc[k]==1 and X_train.iloc[k][j]==1:
				v1y1+=1
		p1[j]=v0y0/count0
		p2[j]=v0y1/count1
		p3[j]=v1y0/count0
		p4[j]=v1y1/count1

	
	return mp0,mp1,p1,p2,p3,p4

def testing(X_test,Y_test,mp0,mp1,p1,p2,p3,p4):
	truep=0
	falsep=0
	falsen=0

	count=0

	size=X_test.shape[0]
	for i in range(X_test.shape[0]):
		cr=X_test.iloc[i]
		y=Y_test.iloc[i]
		prod1=1
		prod2=1
		for j in range(len(cr)):
			if cr[j]==0:
				prod1=prod1*p1[j]
				prod2=prod2*p2[j]
				
			else:
				prod1=prod1*p3[j]
				prod2=prod2*p4[j]
				

		prediction=1
	
		if(prod1>prod2):
			prediction=0

		if(prediction==Y_test.iloc[i]):
			count+=1
	
	return (count/size)








