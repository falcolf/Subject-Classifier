import requests, json

if __name__ == '__main__':
	# Request the model 
	url = "http://13.127.127.229/predict" #url for prediction
	q = str(input("Enter Query : ")) #User Query
	data = json.dumps({'query':q}) #Conversion to JSON . NOTE : DO NOT change json key value
	r = requests.post(url,data) # Make request to url passing the data . NOTE : POST REQUEST ONLY.
	a = r.json() #get json reply
	b = json.JSONEncoder().encode(a) #ENCODE to JSON
	print(b) #PRINT JSON NOTE : subject is passed as single object JSON extractble via key value : 'sub' 
