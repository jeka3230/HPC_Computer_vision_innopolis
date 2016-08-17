import os

import datetime
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def classify(clf):
    images_with_labels = []

    def preprocess_hog(digits):
        samples = []
        for img in digits:
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)
            bin_n = 16
            bin = np.int32(bin_n * ang / (2 * np.pi))
            bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
            mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
            hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
            hist = np.hstack(hists)

            # transform to Hellinger kernel
            eps = 1e-7
            hist /= hist.sum() + eps
            hist = np.sqrt(hist)
            hist /= cv2.norm(hist) + eps

            samples.append(hist)
        return np.float32(samples)

    UseFile = True
    mnist_path = "mnist"
    digits_file_name = "digits.npy"
    if UseFile:
        images_with_labels = np.load(os.path.join(mnist_path, digits_file_name))
    else:
        for num in range(10):
            k = 0
            for (dir_path, dir_names, file_names) in os.walk(os.path.join(mnist_path, str(num))):
                for file_name in file_names:
                    image = cv2.imread(os.path.join(dir_path, file_name), 0)
                    images_with_labels.append((image, num))
                    k+=1
                    if k >= 1000:
                        break
        np.save(os.path.join(mnist_path, digits_file_name), images_with_labels)

    width = images_with_labels[0][0].shape[1]
    height = images_with_labels[0][0].shape[0]

    print images_with_labels.shape

    train_count = 70

    train = []
    test = []

    for i in range(0, len(images_with_labels) - len(images_with_labels) % 100, 100):
        train.extend(images_with_labels[i:i+train_count])
        test.extend(images_with_labels[i+train_count:i + 100])

    train, train_labels = zip(*train)
    test, test_labels = zip(*test)

    train = np.array(train)
    test = np.array(test)

    print('Train size: ' + str(train.shape[0]))
    print('Test size: ' + str(test.shape[0]))

    test_subset = [next(index for index in range(len(test)) if test_labels[index] == num) for num in range(10)]
    #
    # for i in range(10):
    #     plt.subplot(1, 10, i + 1)
    #     plt.imshow(test[test_subset[i]], cmap='gray')
    #     plt.xticks([]), plt.yticks([])
    # plt.show()


    t_start = datetime.datetime.now()

    train_hog = preprocess_hog(np.array((train)))
    print train_hog.shape
    clf.fit(train_hog, train_labels)
    t_elapsed = datetime.datetime.now() - t_start

    print("Training time: " + str(t_elapsed.total_seconds()))

    t_start = datetime.datetime.now()
    res = clf.predict(preprocess_hog(np.array(test)))
    t_elapsed = datetime.datetime.now() - t_start
    correct = np.sum(res==test_labels)
    print (str(correct))
    print(type(clf).__name__ + " " + str(float(correct)/res.size))
    print("Test time: " + str(t_elapsed.total_seconds()))

    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(test[test_subset[i]], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title(res[test_subset[i]])
    plt.show()

# classify(knn())
classify(RandomForestClassifier(n_jobs=4, n_estimators=100000))
# classify(svm.SVC(kernel="rbf", gamma=6, C=2))