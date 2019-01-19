''' feature extraction using librosa '''


# necessary modules
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os,glob


def calculate_features(audio_series):
    
    # calculating energy standard deviation
    frame_length = 882
    RMS_energy = librosa.feature.rmse(audio_series,frame_length=882)
    energy = RMS_energy**2 * frame_length
    energy_std = np.std(energy)     # required feature
    
    # calculating mean value and standard deviation of difference energy
    diff_energy = np.diff(energy)       
    diff_energy_mean = np.mean(diff_energy)     # required feature
    diff_energy_std = np.std(diff_energy)       # required feature
    
    # calculating standard deviation of autocorrelation 
    autoCorrelate = librosa.core.autocorrelate(audio_series)
    autoCorrelate_std = np.std(autoCorrelate)       # required feature
    
    # calculating standard deviation of difference autocorrelation
    diff_autoCorrelate = np.diff(autoCorrelate)
    diff_autoCorrelate_std = np.std(diff_autoCorrelate)     # required feature
    
    # calculating MFCCs
    MFCC = librosa.feature.mfcc(audio_series,sr=22050,n_mfcc=9)
    
    # calculating mean and standard deviation of difference of 9th MFCC
    MFCC_9 = MFCC[8,:]
    diff_MFCC_9 = np.diff(MFCC_9)
    diff_MFCC_9_mean = np.mean(diff_MFCC_9)     # required feature
    diff_MFCC_9_std = np.std(diff_MFCC_9)       # required feature
    
    # calculating mean and standard deviation of difference of 7th MFCC
    MFCC_7 = MFCC[6,:]
    diff_MFCC_7 = np.diff(MFCC_7)
    diff_MFCC_7_mean = np.mean(diff_MFCC_7)     # required feature
    diff_MFCC_7_std = np.std(diff_MFCC_7)       # required feature
    
    # calculating standard deviation of 4th MFCC
    MFCC_4 = MFCC[3,:]
    MFCC_4_std = np.std(MFCC_4)     # required feature
    
    # low short time energy ratio
    energy_mean = np.mean(energy)
    LSTER = np.count_nonzero(energy<0.333*energy_mean) * 100 / np.size(energy)      # required feature
    
    
    features = np.array([energy_std,diff_energy_mean,diff_energy_std,autoCorrelate_std,diff_autoCorrelate_std,diff_MFCC_9_mean,diff_MFCC_9_std,diff_MFCC_7_mean,diff_MFCC_7_std,MFCC_4_std,LSTER])   
    return features



''' loading speech data using librosa '''

path = r"C:\Users\Skanda\Documents\Studies\Machine learning\Music_Speech\speech_wav"
X = np.empty((0,11),dtype=float)

for filename in glob.glob(os.path.join(path, '*.wav')):
    data, sampling_rate = librosa.load(filename)
    duration = int(4 * data.size / 30)  # audio samples 30 sec and duration taken 4 sec
    audio_series = data[0:duration]
    feature_set = calculate_features(audio_series)
    feature_set = feature_set.reshape(1,11)
    X = np.append(X,feature_set,axis=0)
    print("processing: " + filename)
    
y = np.zeros((len(X),1),dtype=int)        # 0s for speech files
    

''' loading music data using librosa '''

path = r"C:\Users\Skanda\Documents\Studies\Machine learning\Music_Speech\music_wav"

for filename in glob.glob(os.path.join(path, '*.wav')):
    data, sampling_rate = librosa.load(filename)
    duration = int(4 * data.size / 30)  # audio samples 30 sec and duration taken 4 sec
    audio_series = data[0:duration]
    feature_set = calculate_features(audio_series)
    feature_set = feature_set.reshape(1,11)
    X = np.append(X,feature_set,axis=0)
    print("processing: " + filename)

y = np.append(y,np.ones((len(X)-len(y),1),dtype=int),axis=0)        # 1s for music


# saving X and y as numpy arrays
np.save("X_values",X)
np.save("y_values",y)
''' files X_values,y_values can be directly loaded as numpy arrays '''

''' End of feature extraction. '''