from sklearn import datasets
from sklearn import svm
import time


if __name__ == '__main__':
    data = datasets.load_iris()
    iris_x = data.data
    iris_y = data.target

    iris_x_train = iris_x[: -10]
    iris_y_train = iris_y[: -10]
    iris_x_test = iris_x[-10:]
    iris_y_test = iris_y[-10:]

    svc = svm.SVC()

    svc.fit(iris_x_train, iris_y_train)
    print(svc.score(iris_x_test, iris_y_test))

    time1 = time.time()

    print(svc.predict(iris_x_test))
    print(iris_y_test)

    time2 = time.time()
    print("func toke %f ms" % ((time2-time1) * 1000))
