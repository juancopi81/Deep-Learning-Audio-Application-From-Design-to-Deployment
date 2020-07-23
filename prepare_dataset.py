# MFCC (# Timestep/segment, 13 # of coefficients)
import librosa
import os
import json

DATASET_PATH = "Dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 # 1 sec worth of sound 

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):

	# Create data dictionary
	data = {
		"mappings": [],
		"labels": [],
		"MFCCs": [],
		"files": []
	}

	# Loop over all subdirs
	for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

		# we need to ensure we are not at root level
		if dirpath is not dataset_path:

			# update the mappings
			category = dirpath.split("/")[-1] # dataset/down -> [dataset, down]
			data["mappings"].append(category)

			# Print
			print(f"Proccesing {category}")

			# loop through all the filenames and extract MFCCs
			for f in filenames:

				# Get the filepath
				file_path = os.path.join(dirpath, f)

				# Load the audiofile
				signal, sr = librosa.load(file_path) 

				# Ensure the audifile is at least 1 second
				if len(signal) >= SAMPLES_TO_CONSIDER:

					# enforce 1 second long signal
					signal = signal[:SAMPLES_TO_CONSIDER]

					# Extract the MFCCs
					MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

					# Store data -1 since Dataset is number zero
					data["labels"].append(i-1)
					data["MFCCs"].append(MFCCs.T.tolist())
					data["files"].append(file_path)
					print(f"{file_path}: {i-1}")

	# Store in json file
	with open(json_path, "w") as fp:
		json.dump(data, fp, indent=4)


if __name__=="__main__":
	prepare_dataset(DATASET_PATH, JSON_PATH)


