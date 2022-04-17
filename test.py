from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import MultiLabelBinarizer


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
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  # Your mom is ugly
    [1, 0, 0, 0, 0, 0, 0, 1, 0],  # Your mom is nice
    [0, 1, 0, 0, 0, 1, 0, 0, 0],  # Your dad is harry
    [0, 0, 1, 1, 0, 0, 0, 0, 0],  # Your grandma is old
    [0, 0, 1, 1, 0, 0, 0, 1, 0],  # Your grandma is old and nice
]

labels = [
    [0],
    [1, 2],
    [1],
    [1],
    [1, 2],
]


print(labels)
labels = MultiLabelBinarizer().fit_transform(labels)
print(labels)
