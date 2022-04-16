import os
import dlib
import numpy as np
import utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('lib/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('lib/dlib_face_recognition_resnet_model_v1.dat')


def get_feature(img):
    dets = detector(img)
    if len(dets):
        d = dets[0]
        l = np.maximum(d.left(), 0)
        u = np.maximum(d.top(), 0)
        r = np.minimum(d.right(), img.shape[1])
        d = np.minimum(d.bottom(), img.shape[0])
        rec = dlib.rectangle(l, u, r, d)
        shape = predictor(img, rec)

        descriptor = facerec.compute_face_descriptor(img, shape)
        feature = np.array(descriptor).reshape((1, 128))

        return feature
    else:
        return -1


def train(class_num):
    feature_dict = np.zeros((class_num, 128))
    train_dir = 'train'
    for i in range(class_num):
        sub_dir = 'ID' + str(i + 1)
        path = os.path.join(train_dir, sub_dir)
        videos = os.listdir(path)
        feature = np.zeros((1, 128))

        samples = int(len(videos))
        for k in range(samples):
            video = videos[k]
            video_path = os.path.join(path, video)
            video_frame, _ = utils.read_video(video_path)
            img = video_frame[0]
            current = get_feature(img)
            if type(current) != int:
                feature += current
            else:
                samples -= 1
        feature /= samples
        feature_dict[i] = feature

    return feature_dict


def test(feature_dict, video):
    diff = np.zeros((feature_dict.shape[0]))
    img = video[0]
    feature = get_feature(img)
    for i in range(feature_dict.shape[0]):
        diff[i] = np.linalg.norm(feature_dict[i] - feature)

    pred = np.argmin(diff) + 1
    return pred
