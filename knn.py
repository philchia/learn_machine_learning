from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    data = datasets.load_iris()
    iris_x = data.data
    iris_y = data.target

    iris_x_train = iris_x[: -10]
    iris_y_train = iris_y[: -10]
    iris_x_test = iris_x[-10:]
    iris_y_test = iris_y[-10:]

    knn = KNeighborsClassifier()
    knn.fit(iris_x_train, iris_y_train)
    print(knn.score(iris_x_test, iris_y_test))

    print(knn.predict(iris_x_test))
    print(iris_y_test)