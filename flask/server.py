import random
import os
from flask import Flask, request, jsonify
from keyword_spotting_service import Keyword_Spotting_Service

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

	# Get audio file and save it
	audio_file = request.files["file"]
	file_name = str(random.randint(0, 100000))
	audio_file.save(file_name)

	# Invoke keyword spotting service
	kss = Keyword_Spotting_Service()

	# Make a prediction
	predicted_keyword = kss.predict(file_name)

	# Remove the audio file
	os.remove(file_name)

	# Send back the predicted Keyword in Json format
	data = {"keyword": predicted_keyword}

	return jsonify(data)

if __name__=="__main__":
	app.run(debug=False)