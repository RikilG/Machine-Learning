import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(trainY, g):
    return np.sum(np.multiply(trainY, np.log(g))+np.multiply((1-trainY), np.log(1-g)))


def logisticRegression(trainX, trainY, testX, testY, noOfIter, alpha, lamda, options):
    if options[1] == "random":
        w = np.random.rand(trainX.shape[1])
    elif options[1] == "normal" or options[1] == "gaussian":
        # Since 99% accuracy has weights around this distribution
        w = np.random.normal(0, 7, trainX.shape[1])
    elif options[1] == "uniform":
        w = np.random.uniform(-7, 7, trainX.shape[1])
    error = []
    x = []
    for i in range(noOfIter):
        z = np.dot(trainX, w)
        g = sigmoid(z)  # Hypothesis
        error.append(cost(trainY, g))
        x.append(i)
        if options[0] == "without":
            delW = (np.dot(np.transpose(trainX), (g-trainY))) / len(trainY)
        elif options[0] == "L1 norm":
            delW = (np.dot(np.transpose(trainX), (g-trainY)) +
                    lamda*np.sign(w)/2) / len(trainY)
        elif options[0] == "L2 norm":
            delW = (np.dot(np.transpose(trainX), (g-trainY)) +
                    lamda*w) / len(trainY)
        w = w - alpha * delW
    plt.plot(x, error)
    plt.show()
    z = np.dot(testX, w)
    test_hyp = sigmoid(z)  # testing hypothesis
    return w, test_hyp >= 0.5


if __name__ == "__main__":
    data = pd.read_csv("./datasets/data_banknote_authentication.txt",
                       sep=',', header=None).values
    p = np.random.permutation(1372)
    size = int(0.8*len(data))
    trainX = data[p[:size], :-1]
    testX = data[p[size:], :-1]
    trainY = data[p[:size], -1]
    testY = data[p[size:], -1]

    trainX = (trainX-np.mean(trainX))/np.std(trainX)
    testX = (testX-np.mean(testX))/np.std(testX)

    # attaching a column of ones
    trainX = np.concatenate((np.ones((trainX.shape[0], 1)), trainX), axis=1)
    # attaching a column of ones
    testX = np.concatenate((np.ones((testX.shape[0], 1)), testX), axis=1)

    # Alpha varied
    print("**********comparison on basis of different factors**********")

    for opt0 in ["without", "L1 norm", "L2 norm"]:
        for opt1 in ["random", "gaussian", "uniform"]:
            for alpha in [0.01, 0.1, 0.3, 1, 3]:
                for lamda in [0.1, 0.5, 1]:
                    w, pred = logisticRegression(
                        trainX, trainY, testX, testY, 5000, alpha, lamda, [opt0, opt1])
                    if opt0 != "without":
                        print("        regularization constant is : ", lamda)
                        print("        regularisation type is: ", opt0)
                    print("        learning rate is : ", alpha)
                    print("        Weight initialisation: ", opt1)
                    print("        Accuracy is : ",
                          (pred == testY).mean()*100, "%")
                    print("        Weight values : ", w)
                    print("\n")
                    if opt0 == "without":
                        break
