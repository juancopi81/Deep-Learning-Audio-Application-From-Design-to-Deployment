import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050 # 1 Second

class _Keyword_Spotting_Service:

	model = None
	_mapping = [
        "go",
        "off",
        "right",
        "down",
        "no",
        "left",
        "on",
        "stop",
        "up",
        "yes"
    ]
	_instance = None

	def predict(self, file_path):

		# Extract the MFCCs
		MFCCs = self.preprocess(file_path) # (# Segments (44), # Coeff(13))

		# Convert 2d MFCCs array into 4d array -> (# Samples, # Segments, # Coeff, 1 -> # Channels for CNN)
		MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

		# Make prediction
		predictions = self.model.predict(MFCCs) # 1 Example -> [ [0.1, 0.6, 0.1, ...] ]
		predicted_index = np.argmax(predictions)
		predicted_keyword = self._mapping[predicted_index]

		return predicted_keyword

	def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
		
		# Load the audio file
		signal, sr = librosa.load(file_path)

		# Ensure consistency in the audio file length
		if len(signal) > NUM_SAMPLES_TO_CONSIDER:
			signal = signal[:NUM_SAMPLES_TO_CONSIDER]

		# Extract the MFCCs
		MFCCs = librosa.feature.mfcc(signal, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

		return MFCCs.T


def Keyword_Spotting_Service():

	# Ensure we only have 1 instance of KSS
	if _Keyword_Spotting_Service._instance is None:
		_Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
		_Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)

	return _Keyword_Spotting_Service._instance


if __name__=="__main__":
	kss = Keyword_Spotting_Service()

	keyword1 = kss.predict("test/down.wav") 
	keyword2 = kss.predict("test/left.wav")

	print(f"Predicted keywords: {keyword1} {keyword2}")

