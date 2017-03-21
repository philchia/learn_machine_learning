from sklearn import datasets
from sklearn import naive_bayes


if __name__ == '__main__':
    data = datasets.load_iris()
    iris_x = data.data
    iris_y = data.target

    iris_x_train = iris_x[: -10]
    iris_y_train = iris_y[: -10]
    iris_x_test = iris_x[-10:]
    iris_y_test = iris_y[-10:]

    nb = naive_bayes.BaseNB()
    nb.fit(iris_x_train, iris_y_train)
    print(nb.score(iris_x_test, iris_y_test))

    print(nb.predict(iris_x_test))
    print(iris_y_test)