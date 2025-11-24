import kagglehub

# Download latest version
path = kagglehub.dataset_download("imsparsh/audio-speech-sentiment")

print("Path to dataset files:", path)