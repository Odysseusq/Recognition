import os
import utils
import pickle
from librosa import resample
import numpy as np
from python_speech_features import mfcc
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture as GMM


def calculate_delta(array):
    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas


def get_feature(audio, fs):
    mfcc_feat = mfcc(audio, fs, 0.025, 0.01, 20, appendEnergy=True)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta_feat = calculate_delta(mfcc_feat)
    feature = np.hstack((mfcc_feat, delta_feat))
    return feature


def train(class_num):
    train_dir = 'train'
    dest_dir = 'lib/'
    models = []
    for i in range(class_num):
        features = np.asarray(())
        sub_dir = 'ID' + str(i + 1)
        path = os.path.join(train_dir, sub_dir)
        wavs = os.listdir(path)
        samples = int(len(wavs))
        for k in range(samples):
            wav = wavs[k]
            wav_path = os.path.join(path, wav)
            audio_trace = utils.read_audio(wav_path, 44100)
            audio_trace = resample(np.squeeze(audio_trace), 44100, 16000)
            vec = get_feature(audio_trace, 16000)
            if features.size == 0:
                features = vec
            else:
                features = np.vstack((features, vec))
        gmm = GMM(n_components=16, max_iter=300, covariance_type='diag', n_init=3)
        gmm.fit(features)
        models.append(gmm)

    pickle.dump(models, open(dest_dir + 'model.out', 'wb'))

    return


def test(audio_trace):
    source_path = 'lib/model.out'
    models = pickle.load(open(source_path, 'rb'))
    speakers = [x for x in range(1, 21)]
    vec = get_feature(audio_trace, 16000)

    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vec))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    ID = speakers[winner]

    return ID
