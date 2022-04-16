import numpy as np
from sklearn.decomposition import FastICA
from librosa import resample
import task1, task2


def pos(video_frames, audio):
    feature_dict = np.load('lib/feature_dict.npy')
    id_video = []
    id_audio = []
    for i in range(3):
        id_video.append(task1.test(feature_dict, video_frames[:, :, 224 * i:224 * (i + 1), :]))
        audio_res = resample(audio[i], 44100, 16000)
        id_audio.append(task2.test(audio_res))

    posed = audio.copy()
    id_audio_2 = id_audio.copy()
    for i in range(2):
        if id_audio[i] != id_video[i]:
            for j in range(3):
                if id_audio[i] == id_video[j]:
                    posed[j] = audio[i]
                    id_audio_2[j] = id_audio[i]
                    posed[i] = audio[j]
                    id_audio_2[i] = id_audio[j]

    return posed


def test(video_frames, audio_trace):
    ica = FastICA(n_components=2, max_iter=300)
    separated = ica.fit_transform(audio_trace)
    rho_0 = np.sum(audio_trace[:, 0] * separated[:, 0]) / np.linalg.norm(separated[:, 0]) / np.linalg.norm(
        audio_trace[:, 0])
    rho_1 = np.sum(audio_trace[:, 0] * separated[:, 1]) / np.linalg.norm(separated[:, 1]) / np.linalg.norm(
        audio_trace[:, 1])
    audio_third = audio_trace[:, 0] - separated[:, 0] * rho_0 - separated[:, 1] * rho_1
    separated = np.column_stack((separated, audio_third))
    separated = np.transpose(separated)
    posed = pos(video_frames, separated)

    return posed
