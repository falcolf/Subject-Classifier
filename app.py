from flask import Flask, abort, jsonify, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
nltk.download('stopwords')
from sklearn.eternals import joblib
from sklearn.naive_bayes import MultinomialNB
cv = CountVectorizer()
cv = joblib.load('cv.joblib.pkl')
classifier = MultinomialNB()
classifier = joblib.load('classifier.joblib.pkl')

app = Flask(__name__)

@app.route('/')
def index():
   return 'Text Classifier for Surfcourse'

@app.route('/predict', methods=['POST'])
def predict():
	# recieved data['query']
	data = request.get_json(force=True)
	ps = PorterStemmer()
	query = data['query']
	query = query.lower()
	query = query.split()
	query = [ps.stem(word) for word in query if not word in set(stopwords.words('english'))]
	mod_query = ' '.join(query)
	query = cv.transform([' '.join(query)])
	prediction = classifier.predict_proba(query.toarray()).T
	classes = classifier.classes_.tolist()
	subjects = [] 
	while(True):
		val = prediction.argmax(axis=0)
		if(prediction[val][0] == 0):
			break
		subjects.append(val)
		prediction[val]=[0]
	subjects = [classes[i[0]] for i in subjects]
	if(len(subjects)>5):
		subjects = subjects[:5]
	return jsonify(sub=subjects , query = mod_query)

if __name__ == '__main__':
	app.run(host='0.0.0.0')


















