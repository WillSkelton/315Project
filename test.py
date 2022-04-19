from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier

# vocabulary:
#   0 Mom
#   1 Dad
#   2 Grandma
#   3 Old
#   4 Harry
#   5 Stupid
#   6 Wonderful
#   7 Nice
#   8 Ugly

#             0     1     2
# Labels: [ Mean, True, Nice]


data = [
    #0, 1, 2, 3, 4, 5, 6, 7, 8
    [1, -1, -1, -1, -1, -1, -1, -1, 1],  # Your mom is ugly
    [1, -1, -1, -1, -1, -1, -1, 1, -1],  # Your mom is nice
    [-1, 1, -1, -1, -1, 1, -1, -1, -1],  # Your dad is harry
    [-1, -1, 1, 1, -1, -1, -1, -1, -1],  # Your grandma is old
    [-1, -1, 1, 1, -1, -1, -1, 1, -1],  # Your grandma is old and nice
]

labels = [
    [0],
    [1, 2],
    [1],
    [1],
    [1, 2],
]

labels = MultiLabelBinarizer().fit_transform(labels)


def processLabels(labels):
    # print(labels)
    labels = MultiLabelBinarizer().fit_transform(labels)
    # print(labels)


def knnClassifier(data, labels):
    # Create Classifier
    knn = KNeighborsClassifier()

    # Train it
    knn.fit(data, labels)

    return knn


def decisionTree(data, labels):
    # Create Classifier
    dt = DecisionTreeClassifier()

    # Train it
    dt.fit(data, labels)

    return dt


def main():

    # Test Case: "Your dad is nice"
    testCase = [[-1, 1, -1, -1, -1, -1, -1, 1, -1]]

    knn = knnClassifier(data, labels)

    # Predict "Your Dad is Nice"
    prediction = knn.predict(testCase)
    print(prediction[0])

    dt = decisionTree(data, labels)
    prediction = dt.predict(testCase)
    print(prediction[0])


main()
