from sklearn import neural_network
from sklearn import datasets
import time


if __name__ == '__main__':
    data = datasets.load_iris()
    iris_x = data.data
    iris_y = data.target

    iris_x_train = iris_x[: -10]
    iris_y_train = iris_y[: -10]
    iris_x_test = iris_x[-10:]
    iris_y_test = iris_y[-10:]

    nn = neural_network.MLPClassifier()
    nn.fit(iris_x_train, iris_y_train)
    print(nn.score(iris_x_test, iris_y_test))
    time1 = time.time()

    print(nn.predict(iris_x_test))
    print(iris_y_test)

    time2 = time.time()
    print("func toke %f ms" % ((time2-time1) * 1000))