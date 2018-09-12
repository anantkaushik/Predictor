from django.shortcuts import render
from django.contrib.auth import authenticate
import mongoengine,json, pymongo, pandas, pickle,  numpy as np, pandas as pd,matplotlib.pyplot as plt
from pymongo import MongoClient
from bson.json_util import dumps
from collections import OrderedDict
from datetime import date
# Create your views here.
client = MongoClient('mongodb://anant:anant28@localhost:27017')
db = client.testdb
client1 = MongoClient('mongodb://anant:anant28@localhost:27017')
db1 = client.userdb
def home(request):
	c_name = db.collection_names(include_system_collections=False)
	db_name = {'dbName':c_name}
	return render(request, 'testapp/index.html',db_name)

def test(request):    
	result = db.Salary_Data.find()
	column_name = set()
	for i in result:
		for j in i:
			if j != "_id":
				column_name.add(j)
	variables = {'column_name':column_name}
	return render(request, 'testapp/display.html',variables)

def displaydata(request):
	coll = db1.userdata
	coll.update_one({'_id':'5b614a9bcbe020345574e8de'},
		{ '$set': {'d_v' : request.POST['source'],'id_v':request.POST.getlist('cblist')[0]}},
	upsert = True)
	results = db.Salary_Data.find()
	"""for i in result:
		a = []
		for j in i:
			if j != "_id":
				print(i[j])
				a.append(j)"""
	a = []
	for i in results:
		a.append(i)
	df = pd.DataFrame(a)
	dataSet = df.iloc[:,[1,0]].values #dataSet
	variables = {'dataset':dataSet}
	return render(request, 'testapp/displaydata.html',variables)

def setpara(request):
	return render(request, 'testapp/para.html')

def calculate_algo(request):
	ts=request.POST['training_size']
	rs = request.POST['random_state']
	fi = request.POST['fit_intercept']
	coll = db1.userdata
	coll.update_one({'_id':'5b614a9bcbe020345574e8de'},
		{ '$set': {'t_size' : ts, 'r_state': rs, 'f_i':fi} },
	upsert = True)
	print("********",(int(ts)/10),rs)
	rs = int(rs)
	results = db.Salary_Data.find()
	r = db1.userdata.find()
	d__v = ""
	id__v = ""
	for i in r:
		d__v = i['d_v']
		id__v = i["id_v"]
	a = []
	for i in results:
		a.append(i)
	df = pd.DataFrame(a)
	dataSet = df.iloc[:,[1,0]].values #dataset
	x = df.iloc[:,df.columns.get_loc(id__v)].values #dependent variale
	y = df.iloc[:,df.columns.get_loc(d__v)].values #independent varibale
	x = np.reshape(x, (-1, 1))
	y = np.reshape(y, (-1, 1))
	# Fitting Simple Linear Regression to the dataset
	if rs == 0:
		from sklearn.cross_validation import train_test_split
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = (1-(int(ts)/100)))
	else:
		from sklearn.cross_validation import train_test_split
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = (1-(int(ts)/100)), random_state = int(rs))
	# Fitting Simple Linear Regression to the Training set
	from sklearn.linear_model import LinearRegression
	if fi == "True" or fi  == "true":
		regressor = LinearRegression(fit_intercept=True)
	else:
		regressor = LinearRegression(fit_intercept=False)
	regressor.fit(x_train, y_train)
	#Saving the model
	list_pickle_path = 'static/linear_model.pkl'
	list_pickle = open(list_pickle_path, 'wb')
	pickle.dump(regressor, list_pickle)
	list_pickle.close()
	# Predicting the Test set results.
	y_pred = regressor.predict(x_test)
	# Visualising the Test set results
	plt.scatter(x_test, y_test, color = 'red')
	plt.plot(x_train, regressor.predict(x_train), color = 'blue')
	plt.title("Salary vs Experience")
	plt.xlabel(id__v)
	plt.ylabel(d__v)
	plt.savefig("static/test1.png")
	plt.clf()
	from sklearn.metrics import r2_score
	aa = r2_score(y_test, y_pred) 
	variables = {'x_test':x_test,'y_pred':y_pred,'accuracy':aa,'dv':d__v,'idv':id__v}
	return render(request, 'testapp/displayresult.html',variables)

def predd(request):
	r = db1.userdata.find()
	d__v = ""
	id__v = ""
	for i in r:
		d__v = i['d_v']
		id__v = i["id_v"]
	variables = {'dv':d__v,'idv':id__v}
	return render(request, 'testapp/predict.html',variables)

def f_result(request):
	idvv = request.POST['input_idv']
	list_pickle_path = 'static/linear_model.pkl'
 
	# load the unpickle object into a variable
	with open(list_pickle_path, 'rb') as file:  
		pickle_model = pickle.load(file)
	ans = pickle_model.predict([[int(idvv)]])
	coll = db1.userdata
	coll.update_one({'_id':'5b614a9bcbe020345574e8de'},
		{ '$set': {'predicted_dv' : ans[0][0],'independent_variable':idvv} },
	upsert = True)
	d__v = ""
	id__v = ""
	r = db1.userdata.find()
	for i in r:
		d__v = i['d_v']
		id__v = i["id_v"]
	variables = {'idv':idvv,'dv':ans[0][0],'d_v':d__v,'id_v':id__v}
	return render(request, 'testapp/final.html',variables)