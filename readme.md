# Comment Classifier

## Contents

- [Comment Classifier](#comment-classifier)
  - [Contents](#contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Motivation](#11-motivation)
    - [1.2 Questions We Wish to Answer](#12-questions-we-wish-to-answer)
    - [1.3 Challenges](#13-challenges)
    - [1.4 Results](#14-results)
  - [2. Data Mining Task](#2-data-mining-task)
    - [2.1 Task Description](#21-task-description)
    - [2.2 Questions To Answer](#22-questions-to-answer)
    - [2.3 Challenges](#23-challenges)
  - [3. Technical Approach](#3-technical-approach)
    - [3.1 Algorithm](#31-algorithm)
    - [3.2 Addressing Challenges](#32-addressing-challenges)
  - [4. Evaluation Methodology](#4-evaluation-methodology)
    - [4.1 Dataset and Source](#41-dataset-and-source)
    - [4.2 Metrics](#42-metrics)
  - [5. Results and Discussion](#5-results-and-discussion)
    - [5.1 Results](#51-results)
    - [5.2 What Worked](#52-what-worked)
    - [5.3 What Didn't](#53-what-didnt)
  - [6. Retrospective](#6-retrospective)
    - [6.1 Lessons Learned](#61-lessons-learned)
    - [6.2 Hindsight](#62-hindsight)
  - [7. Acknowledgements](#7-acknowledgements)

## 1. Introduction

### 1.1 Motivation

Anyone who's been in an online comment section has seen first-hand what happens when people can say whatever they want with impunity. Sure, there are moderators and they definitely help the situation. However, a volunteer moderator can only do so much and the sheer volume of toxicity can overwhelm the system.

Enter Machine Learning. Computers are already incredible at automating tasks very fast. What if we could write a program that got better at it's job the more it did it? That's exactly what Machine Learning is.

The goal of this project is to take real comments from Wikipedia that have been labeled with varying levels of toxicity and use them to teach a computer what it means for a comment to be "toxic" and then label new comments with a certain degree of accuracy.

### 1.2 Questions We Wish to Answer

1. Can we use Machine Learning to determine the toxicity of a comment?
2. How do we get training and testing data?
3. Which ML Classifiers are best suited for this task?

### 1.3 Challenges

1. Creating the Algorithm or finding a pre-made Multi-label classifier
2. Finding training and testing data
3. Parsing the data
4. Working with huge data sets with reasonable time restraints

### 1.4 Results

1. Decision tree classifier from Scikit Learn package worked best of the ones we had access to.
2. Testing accuracy was about 85%.
3. The sweet-spot we found with training data was about 4000 training samples.

## 2. Data Mining Task

### 2.1 Task Description

- Clearly describe all the details of the task. What is the input data? What is the output of data mining approach? Give examples to illustrate them.

### 2.2 Questions To Answer

- List all the data mining questions that you set out to investigate in this project.

### 2.3 Challenges

- List the key challenges to solve this task

## 3. Technical Approach

### 3.1 Algorithm

The algorithm itself is fairly simple:

1.  Gather a small subset of Training Data and Labels
2.  Import a few classifiers from Scikit Learn python package
3.  Train classifiers on data set
4.  Test classifiers with testing data to determine accuracy
5.  Pick the best one and repeat steps 1-4 with a larger training set.

Luckily, using the classifiers was the more straightforward task. Here is the code for that:

```py
def CreateAndTrainDecisionTree(data, labels):
    # Create Classifier
    dt = DecisionTreeClassifier()

    # Train Classifier
    dt.fit(data, labels)

    # return classifier
    return dt
```

This function returns a model object. To use it, all we have to do is this:

```py
dt = CreateAndTrainDecisionTree(trainingData, trainingLabels)

accuracy = dt.score(testData, testLabels)
```

Configuring the classifiers was the easy part. The tricky part was parsing the data.

The way we did this was to first create a collection of comments. Then, create a sorted set of all words in all comments with no repeats. These would act as our features for our data. We also removed words like "it's" and "about" that don't contribute much to the comment and converted everything to lower case in order to cut down on duplicates and unnecessary features.

Once we had our vocabulary, we could go through each comment and create an array that had a 1 for every vocabulary word that the comment had and a 0 for every one it didn't. This plus an array of labels was our data.

### 3.2 Addressing Challenges

The first challenge to solve was to find data. Luckily, the website Kaggle had [this data set](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview) all ready for us to use.

The next challenge was to find a classifier that supported multilabel classification. In theory, we could have written our own from scratch. However, our time was limited so we searched for a pre-made option. Luckily, the Python package Scikit Learn had exactly what we needed.

After we installed the package, we came up with a very small test data set of our own. It was set up to be solved the same way as our actual task but instead of huge comments with no restrictions, we made a small set of comments with phrases like "Your mom is ugly" or "Your grandma is old" and labelled them Mean, True, or Nice. This way, our problem domain was still a matter of labelling sentences with one or many labels but we had a smaller data set.

Once we had a little sample data set to work with, we started plugging it into different Scikit Learn classifiers that supported multilabel classification. We tried the K-Nearest Neighbor and the Decision Tree and found that the Decision Tree had the highest level of accuracy.

The last step was to parse the Wikipedia data and run that through our Decision Tree. We quickly found out that we didn't have enough space to train on EVERY training comment. There are tens of thousands of comments in the training set and and my computer ran out of RAM trying to store them all in an array. So we tried a few different sizes for subsets and found that 4000 training comments was the sweet-spot.

Once the training data was parsed, all we had to do was train the classifier with it, parse the testing data the same way we parsed the training data, and then test how well our classifier could predict the testing data.

## 4. Evaluation Methodology

### 4.1 Dataset and Source

Like we mentioned above, We were able to get our training and testing data from [Kaggle.com](Kaggle.com). It came in the form of 3 `.csv` files:

- `train.csv`
- `test_labels.csv`
- `test.csv`

Each one was a little different. `train.csv` had the id, the comment, and then the labels on each line. Then, `test_labels.csv` and `test.csv` were similar but the labels were in their own file, presumably to keep testing data and labels separate so the algorithm can't cheat.

The trickiest part of this data was the file size and the fact that it was a `csv` file but the comments could have multiple newlines. Luckily, the python `csvreader` was able to handle the newlines and we just read the file line by line instead of reading it all at once.

### 4.2 Metrics

The metric we used was accuracy. The main reason being that we're not doing anything so complicated that we needed something more advanced. Our goal was to see how well an algorithm could classify a comment as toxic and accuracy showed us that it could do it about 85% of the time.

## 5. Results and Discussion

### 5.1 Results

Our results showed that we can train a decision tree to determine whether or not a Wikipedia comment is toxic. Like we mentioned earlier, our decision tree was able to predict with about 85% accuracy which is pretty good considering that we did not use the entire training data set and we are new to machine learning. I bet that with further optimization and more experience with classifiers such as deep networks, we could achieve an even higher accuracy.

### 5.2 What Worked

The decision tree was the right way to go in terms of classifiers. The main drawback of decision trees is that they can get arbitrarily large which can lead to overfitting. In our case, overfitting didn't seem to be a huge issue and, while the tree does get fairly large, it's not so large that current computer hardware has an issue generating it.

Another thing we did that was fairly successful was using a smaller subset of the training data. We found that the sweet-spot was about 4000 training comments. Since our algorithm will create a vocabulary of each unique word from the training data and use the existence of each word in each comment as a feature, the more comments we use for training, the larger a tree is and the longer it takes to train. 4000 training points provided a nice balance between training time/tree size and accuracy. IT only took a minute or two on my laptop to train.

### 5.3 What Didn't

The main thing that didn't work was trying to use all the training data. The problem was that the way we are loading it into the classifier is simply storing it in a two-dimensional array and we quickly ran out of RAM trying to allocate space for an array that big.

## 6. Retrospective

### 6.1 Lessons Learned

The biggest takeaway from this project was learning how multilabel classification works and learning how to select the right classifier for a given task. In addition, wo gor a good deal of practice working with large datasets.

### 6.2 Hindsight

The main thing we'd do better next time would to find a better way to store the training data so we can use the entire training set.

## 7. Acknowledgements

- [Scikit Learn Documentation](https://scikit-learn.org/stable/modules/multiclass.html)
