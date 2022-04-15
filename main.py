from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import MultiLabelBinarizer


data, labels = make_multilabel_classification(
    n_samples=5,
    random_state=0,
    return_indicator=False
)

newLabels = MultiLabelBinarizer().fit_transform(labels)

print(newLabels)

knn = KNeighborsClassifier()

knn.fit(data, newLabels)
