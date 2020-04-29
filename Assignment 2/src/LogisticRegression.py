import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(trainY, g):
    return np.sum(np.multiply(trainY, np.log(g))+np.multiply((1-trainY), np.log(1-g)))/trainY.shape[0]


k = 0
j = 0
fig, axs = plt.subplots(1)


def logisticRegression(trainX, trainY, crossvalX, crossvalY, noOfIter, lamda, alpha=0.2, options=["without", "uniform"]):
    if options[1] == "random":
        w = np.random.rand(trainX.shape[1])
    elif options[1] == "normal" or options[1] == "gaussian":
        # Since 99% accuracy has weights around this distribution
        w = np.random.normal(0, 1, trainX.shape[1])
    elif options[1] == "uniform":
        w = np.random.uniform(-1, 1, trainX.shape[1])

    w = np.array([1, 0, 1, 0, 1])
    errorX = []
    errorY = []
    x = []
    best = -1
    #global j, k, fig, axs
    fig, axs = plt.subplots(1)
    for i in range(noOfIter):
        z = np.dot(trainX, w)
        g = sigmoid(z)  # Hypothesis
        errorX.append(abs(cost(trainY, g)))
        errorY.append(abs(cost(crossvalY, sigmoid(np.dot(crossvalX, w)))))
        if best == -1 and abs(cost(trainY, g)) < 0.15:
            best = i
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

    axs.plot(x, errorX)
    axs.plot(x, errorY)
    axs.set_xlim([0, noOfIter+5])
    axs.set_ylim([0, 5])
    axs.set_xlabel('Iterations')
    axs.set_ylabel('Cost')
    axs.axvline(x=best, linestyle='dashed')
    plt.show()
    '''
    j += 1
    if j == 5:
        j = 0
        plt.show()
        fig, axs = plt.subplots(5)
    k += 1
    if k == 2:
        k = 0
        j += 1
    '''
    z = np.dot(crossvalX, w)
    test_hyp = sigmoid(z)  # testing hypothesis
    return w, test_hyp >= 0.5


if __name__ == "__main__":
    data = pd.read_csv("./../datasets/data_banknote_authentication.txt",
                       sep=',', header=None).values
    p = np.random.permutation(1372)
    size = int(0.6*len(data))
    size2 = int(0.2*len(data))
    trainX = data[p[:size], :-1]
    trainY = data[p[:size], -1]
    crossvalX = data[p[size:size+size2], :-1]
    crossvalY = data[p[size:size+size2], -1]
    testX = data[p[size+size2:], :-1]
    testY = data[p[size+size2:], -1]

    trainX = (trainX-np.mean(trainX))/np.std(trainX)
    crossvalX = (crossvalX-np.mean(crossvalX))/np.std(crossvalX)
    testX = (testX - np.mean(testX))/np.std(testX)
    # attaching a column of ones
    trainX = np.concatenate((np.ones((trainX.shape[0], 1)), trainX), axis=1)
    # attaching a column of ones
    crossvalX = np.concatenate(
        (np.ones((crossvalX.shape[0], 1)), crossvalX), axis=1)

    testX = np.concatenate(
        (np.ones((testX.shape[0], 1)), testX), axis=1)

    # Alpha varied
    print("**********comparison on basis of different factors**********")

    alpha = 0.2
    opt1 = "gaussian"
    lamda = 0
    opt0 = "without"
    # for opt0 in ["L1 norm", "L2 norm", "without"]:
    # for opt1 in ["random", "gaussian", "uniform"]:
    # for alpha in [0.001, 0.01, 0.05, 0.1, 0.2]:
    # for lamda in [0.0001, 0.001, 0.01, 0.1, 2]:
    w, pred = logisticRegression(
        trainX, trainY, crossvalX, crossvalY, 250, lamda, alpha, [opt0, opt1])
    if opt0 != "without":
        print("        regularization constant is : ", lamda)
        print("        regularisation type is: ", opt0)
    print("        learning rate is : ", alpha)
    print("        Weight initialisation: ", opt1)
    print("        Accuracy is : ",
          (pred == crossvalY).mean()*100, "%")
    print("        Weight values : ", w)
    print("\n")
    '''
    if opt0 == "without":
        break
        '''

    alpha = 0.2
    opt1 = "gaussian"
    lamda = 0.1
    opt0 = "L2 norm"
    # TESTING
    w, pred = logisticRegression(
        trainX, trainY, testX, testY, 250, lamda, alpha, [opt0, opt1])
    if opt0 != "without":
        print("        regularization constant is : ", lamda)
        print("        regularisation type is: ", opt0)
    print("        learning rate is : ", alpha)
    print("        Weight initialisation: ", opt1)
    print("        Accuracy is : ",
          (pred == testY).mean()*100, "%")
    print("        Weight values : ", w)
    print("\n")
    '''
    if opt0 == "without":
        break
    '''
