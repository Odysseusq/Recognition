"""
视听信息系统导论大作业
秦圣岭 彭熙然 邴格格

特点：
方法较为简单但非常高效，不使用深度学习方法，全部使用CPU计算，完全不需要pytorch/tensorflow和GPU
使用的额外的库很少，只使用了dlib、python_speech_features、sklearn等用于提取特征
运行时间非常短，运行一次测试只需要约半分钟，Task1和Task2在5秒左右就能完成测试，Task3也只需要20秒左右

方法：
基本上来源于课程中讲过的方法。
Task1使用dlib库提取人脸特征，再寻找与测试图片特征最相近的类别。
Task2使用MFCC提取语音的梅尔谱特征，再使用GMM聚类。
Task3使用ICA独立成分分析，将两个声道作为输入，即可盲分离出独立的语音。再结合Task1、2的结果，就可以匹配说话人。

文件说明：
task1/2/3.py：处理三个任务的程序
lib文件夹：依赖库
lib/feature_dict.npy：提取到的人脸特征
model.out：提取到的语音特征
*.dat：人脸特征和位置提取的依赖库

函数接口：
在import tast1/2/3以后，可以使用task1/2/3.test()来运行测试
task1.test(feature_dict, video_frames)：输入特征向量集与视频，输出识别的说话人ID，类型int
task2.test(audio_trace)：输入音频，输出识别的说话人ID，类型int
task3.test(video_frames, audio_trace)：输入视频与原音频，输出3*n的音频，行号0/1/2分别对应左中右，类型np.ndarray

结果：
Task1: 0.94
Task2: 0.88
Task3: SISDR_blind 最佳8.2，平均4~8，SISDR_match 最佳7.3，平均3~7

注：Task3的结果不稳定，如果结果较低，麻烦助教重新运行一下～

我们保证这个结果能简单复现。我们很高兴，能用非深度学习的方法较好地完成任务。方法基本来自课堂，感谢老师和助教的付出！
"""

import numpy as np
import soundfile as sf
import os, json
import utils
import nussl
import task1, task2, task3


def test_task1(video_path):
    # 测试1
    result_dict = {}
    class_num = 20
    print('\nTask 1 is running...')

    # 训练过程，单纯测试中不需要，如果需要重新提取特征，则取消注释
    # feature_dict = task1.train(class_num)
    # np.save('lib/feature_dict.npy', feature_dict)

    # 加载已提取出来的特征
    feature_dict = np.load('lib/feature_dict.npy')

    for file_name in os.listdir(video_path):
        ## 读取MP4文件中的视频,可以用任意其他的读写库
        video_frames, video_fps = utils.read_video(os.path.join(video_path, file_name))

        ## 返回一个ID
        result_dict[file_name] = utils.ID_dict[task1.test(feature_dict, video_frames)]

    return result_dict


def test_task2(wav_path):
    # 测试2
    result_dict = {}
    class_num = 20

    # 训练过程，单纯测试中不需要，如果需要重新提取特征，则取消注释
    # task2.train(class_num)

    print('\nTask 2 is running...')
    for file_name in os.listdir(wav_path):
        ## 读取WAV文件中的音频,可以用任意其他的读写库
        audio_trace = utils.read_audio(os.path.join(wav_path, file_name), sr=16000)

        ## 返回一个ID
        result_dict[file_name] = utils.ID_dict[task2.test(audio_trace)]

    return result_dict


def test_task3(video_path, result_path):
    # 测试3
    if os.path.isdir(result_path):
        print('warning: using existed path as result_path')
    else:
        os.mkdir(result_path)

    print('\nTask 3 is running...')

    for file_name in os.listdir(video_path):
        ## 读MP4中的图像和音频数据，例如：
        idx = file_name[-7:-4]  # 提取出序号：001, 002, 003.....

        video_frames, video_fps = utils.read_video(os.path.join(video_path, file_name))
        audio_trace = utils.read_audio(os.path.join(video_path, file_name), sr=44100)

        ## 做一些处理
        audio = task3.test(video_frames, audio_trace)

        audio_left = audio[0]
        audio_middle = audio[1]
        audio_right = audio[2]

        ## 输出结果到result_path
        sf.write(os.path.join(result_path, idx + '_left.wav'), audio_left, 44100)
        sf.write(os.path.join(result_path, idx + '_middle.wav'), audio_middle, 44100)
        sf.write(os.path.join(result_path, idx + '_right.wav'), audio_right, 44100)


if __name__ == '__main__':
    ## testing task1
    with open('./test_offline/task1_gt.json', 'r') as f:
        task1_gt = json.load(f)
    task1_pred = test_task1('./test_offline/task1')
    task1_acc = utils.calc_accuracy(task1_gt, task1_pred)
    print('accuracy for task1 is:', task1_acc)

    ## testing task2
    with open('./test_offline/task2_gt.json', 'r') as f:
        task2_gt = json.load(f)
    task2_pred = test_task2('./test_offline/task2')
    task2_acc = utils.calc_accuracy(task2_gt, task2_pred)
    print('accuracy for task2 is:', task2_acc)

    # testing task3
    test_task3('./test_offline/task3', './test_offline/task3_estimate')
    task3_SISDR_blind = utils.calc_SISDR('./test_offline/task3_gt', './test_offline/task3_estimate',
                                         permutaion=True)  # 盲分离
    print('strength-averaged SISDR_blind for task3 is:', task3_SISDR_blind)
    task3_SISDR_match = utils.calc_SISDR('./test_offline/task3_gt', './test_offline/task3_estimate',
                                         permutaion=False)  # 定位分离
    print('strength-averaged SISDR_match for task3 is: ', task3_SISDR_match)
