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
  - [6. Lessons Learned](#6-lessons-learned)
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

Comment sections on the internet can be very toxic and our goal is to train a machine learning algorithm to determine whether or not a comment is toxic.

### 2.2 Questions To Answer

- List all the data mining questions that you set out to investigate in this project.

### 2.3 Challenges

- List the key challenges to solve this task

## 3. Technical Approach

### 3.1 Algorithm

- Describe all the details of your algorithmic approach to solve this data mining task and/or answering the data mining questions.
- An algorithmic pseudo-code and/or a figure (block diagram) to illustrate the approach will be good.

### 3.2 Addressing Challenges

- How are you addressing the challenges mentioned above

## 4. Evaluation Methodology

### 4.1 Dataset and Source

- Explain the dataset and its source that you employed to study this task. Any specific challenges to use this data for your task.

### 4.2 Metrics

- List the metrics you employed to evaluate the output of data mining task and/or questions investigated. Justify their choice from real-world applications perspective.

## 5. Results and Discussion

### 5.1 Results

- Present and explain results in a step-by-step manner to tell us a story about what you have discovered by doing this project (all graphs and tables should be properly labeled with legends and captions. they should be self-sufficient to understand the results)

### 5.2 What Worked

- What worked and why?

### 5.3 What Didn't

- What didn't work and why not?

## 6. Lessons Learned

- What did you learn by doing this project? In the hindsight, would you have made some different decisions to improve the project further?

## 7. Acknowledgements

- Acknowledge all the sources of help you got to do this project
