import numpy as np

# labels = np.load("/home/yunus/projects/vad_using_one_shot_learning/dataset/IPAD_dataset/R01/test_label/014.npy")
# print(labels.shape)
# print(np.unique(labels))
# print()
# print(labels)

path = "/home/yunus/projects/vad_using_one_shot_learning/dataset/IPAD_dataset/R01/test_label/0"
pathnpy = ""

# print("/home/yunus/projects/vad_using_one_shot_learning/data/results/anomaly_scores_R01.npy")
# labelsx = np.load("/home/yunus/projects/vad_using_one_shot_learning/data/results/anomaly_scores_R01.npy")
# print(labelsx.shape)
# print(np.unique(labelsx))
# print(labelsx)
# print("-"*70)

for i in range(1, 16):
    if i < 10:
        pathnpy = path + "0" + str(i) + ".npy"
    else:
        pathnpy = path + str(i) + ".npy"

    print(pathnpy)
    labels = np.load(pathnpy)
    print(labels.shape)
    print(np.unique(labels))
    print(labels)
    print("-"*70)
