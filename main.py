import csv
from datetime import datetime
from doctest import testfile
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


def parseData(filename):
    log(f"Opening file: `{filename}`")
    file = open(filename, encoding='utf-8')
    csvreader = csv.reader(file)

    # extracting field names through first row
    fields = next(csvreader)
    comments = []
    labels = []

    log("Traversing file...")
    # extracting each data row one by one
    for row in csvreader:
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


if __name__ == "__main__":

    numberOfTrainingComments = 500

    stopList = parseStopList()
    comments, trainLabels = parseData("data/train.csv")
    vocabulary = generateVocabulary(comments, stopList)
    trainData = generateDataFromComments(comments, vocabulary)

    log(f"Number of Comments `{len(comments)}`")
    log(f"Number of Data Points `{len(trainData)}`")

    dt = CreateAndTrainDecisionTree(trainData, trainLabels)

    testComments = [
        "Yo bitch Ja Rule is more succesful then you'll ever be whats up with you and hating you sad mofuckas...i should bitch slap ur pethedic white faces and get you to kiss my ass you guys sicken me. Ja rule is about pride in da music man. dont diss that shit on him. and nothin is wrong bein like tupac he was a brother too...fuckin white boys get things right next time.,",

        "fuck islam , fuck Saud -Arabia",
    ]

    testComments = generateDataFromComments(testComments, vocabulary)

    trainLabels = [
        [-1, -1, -1, -1, -1, -1],
        [1, 1, 1, 0, 1, 1]
    ]

    log("Predicting...")
    predictions = dt.predict(testComments)

    log("Predicted Values:")
    for actual, prediction in zip(trainLabels, predictions):
        log(f"{actual}")
        log(f"{prediction}")
        log("----------------------------")
