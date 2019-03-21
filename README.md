# IET-MusicSpeechClassifier
The aim of this project is to build various models to classify a given audio input as a music or speech file. The dataset used was obtained from Marsyas. It contains 64 samples of each, speech and music. The mentioned features were extracted from the audio files using librosa. Scipy was used to build the models. The parameters of the model were fine tuned to get the best results.<br />
Dataset used: "http://marsyas.info/downloads/datasets.html". <br />
Research Paper Referred: "https://link.springer.com/article/10.1155/2009/239892".

### Features extracted: 
1. Standard deviation of energy.
2. Mean value and standard deviation of difference energy.
3. Standard deviation of autocorrelation.
4. Standard deviation of autocorrelation difference.
5. Mean and standard deviation of difference of 9th, 7th, 4th Mel Frequency Cepstrum Coefficients.
6. Low Short time Energy ratio

### Classification Models
1. K-Nearest Neighbour
2. Decision Tree
3. SVC (kernel: linear)
4. SVC (kernel: rbf)
5. Logistic Regression
6. Naive Bayes
7. Ensemble-Random Forest

### Libraries and tools
1. numpy for array related operations and pandas.
2. scikit for built in models.
3. librosa 
4. spyder

### Project Members
1. Bhargav S (Mentor)
2. Skanda U
3. Rahul Gite
4. Abhishek Ranjan
