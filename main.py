import csv
from datetime import datetime
from doctest import testfile
from numpy import number
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier


def log(message):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}]: {message}")


def parseStopList():
    file = open("data/stoplist.txt")

    stopList = {}
    for line in file:
        word = line.strip().lower()
        stopList[word] = True

    return stopList


def parseTrainData(filename, maxLength):
    log(f"Opening file: `{filename}`")
    file = open(filename, encoding='utf-8')
    csvreader = csv.reader(file)

    # extracting field names through first row
    fields = next(csvreader)
    comments = []
    labels = []

    log("Traversing file...")
    # extracting each data row one by one
    for index, row in enumerate(csvreader):
        if index == maxLength:
            break

        if(len(row) < 8):
            continue

        comments.append(row[1])
        labels.append(list(map(lambda num: 1 if num == '1' else -1, row[2:])))

    log("Done")
    return comments, labels


def generateVocabulary(comments, stopList):
    log("Generating Vocabulary...")

    vocabulary = []

    for comment in comments:
        words = comment.lower().strip().split()
        for word in words:
            vocabulary.append(word)

    log(f"Vocabulary has `{len(vocabulary)}` words")

    log("Deduping and sorting vocabulary...")
    vocabulary = sorted(filter(lambda word: word not in stopList, set(vocabulary)))

    log("Removing stop words")

    log(f"Vocabulary has `{len(vocabulary)}` *unique* words")

    return vocabulary


def generateDataFromComments(comments, vocabulary):
    log("Generating Data from Comments...")
    data = []

    for index, comment in enumerate(comments):
        if index % 200 == 0:
            log(f"  - Parsing comment {index+1}/{len(comments)}")

        commentWords = {}

        for word in comment.strip().split():
            commentWords[word] = True

        row = []

        for word in vocabulary:
            row.append(1 if word in commentWords else -1)

        data.append(row)
    log("Done")
    return data


def CreateAndTrainDecisionTree(data, labels):
    # Create Classifier
    log("Creating Decision Tree")
    dt = DecisionTreeClassifier()
    log("Done")
    # Train Classifier
    log("Training Decision Tree")
    dt.fit(data, labels)
    log("Done")
    # return it
    return dt


def parseTestData(filename, maxLength):
    file = open(filename, encoding='utf-8')

    csvreader = csv.reader(file)

    fields = next(csvreader)

    comments = []

    # extracting each data row one by one
    for index, row in enumerate(csvreader):
        if index == maxLength:
            break
        if(len(row) < 2):
            continue

        comments.append(row[1])

    return comments


def parseTestLabels(filename, maxLength):
    file = open(filename, encoding='utf-8')

    csvreader = csv.reader(file)

    fields = next(csvreader)

    labels = []

    # extracting each data row one by one
    for index, row in enumerate(csvreader):
        if index == maxLength:
            break
        labels.append(list(map(lambda num: 1 if num == '1' else -1, row[1:])))

    return labels


if __name__ == "__main__":
    trainLength = 4000
    testlength = 2000

    stopList = parseStopList()
    comments, trainLabels = parseTrainData("data/train.csv", trainLength)
    vocabulary = generateVocabulary(comments, stopList)
    trainData = generateDataFromComments(comments, vocabulary)

    log(f"Number of Comments `{len(comments)}`")
    log(f"Number of Data Points `{len(trainData)}`")

    dt = CreateAndTrainDecisionTree(trainData, trainLabels)

    log("Gathering Test Data...")
    testData = generateDataFromComments(
        parseTestData("data/test.csv", testlength),
        vocabulary
    )

    log("Gathering Test Labels...")
    testLabels = parseTestLabels("data/test_labels.csv", testlength)

    log("Calculating Decision Tree Accuracy...")
    log(f"Accuracy: {dt.score(testData, testLabels)}")
