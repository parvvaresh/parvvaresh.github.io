# sentiment analysis from scratch python for persian tweets

In this model, only 2 libraries are used:

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

We trained our model using datasets collected from Twitter that have **"sad"** and **"joy"** tags
This dataset includes a total of 64,000 tweets, all of which have tags


This repository contains 4 files which are:
1.  load_file.py
2.  Preprocessing.py
3.  model.py
4.  main.py


## 1-load_file.py
In this file, first we read the file, then we separate the text column and the goal, and then replace the **"sad"** with zero and the **"joy"** with 1.



## 2-Preprocessing.py
Pre-processing order:
1.  Removing vowels from the text
2.  Remove all writing from the text
3.  Removing non-Persian words from the text
4.  Tkenize sentences
5.  Tokenize sentences
6.  Remove stop words from tokens
7.  Finding the roots of words

## 3.model.py

Here we implemented 3 classes:
1.  LogisticRegression     
     
            ```
            
            Our model is a logistic regression that categorizes tweets in a binary way.
            The model is first trained with the training data, and then the prediction function is for prediction, and the   accuracy function is written in the class itself.
            
            ```
2.  extraxt_feature

            ```
            
            We create a list of unique words in the entire data set.
            
            ```
            
3.  matrix_features

            ```
            
            Here, she has created a data frame whose index is the unique words and two columns, "freq positive" and "freq negetive", which indicate how often these commas are repeated in sentences with "positive" and "negative" labels, respectively.
            
            ```
4.  set_features

            ```
            
            Now, for each vector, the sum of frequencies with positive labels and the sum of frequencies with negative labels are considered as 2 features for vectors.
            
            ```
## 4.main.py

In the main file, respectively:
1.  Upload file
2.  Data preprocessing
3.  Creating a feature matrix
4.  Dividing data into training and testing
5.  Model training using training data
6.  Model testing using test data
7.  Calculate the accuracy of the model (**98 percent** which is excellent)


The photo below is the output of the work :‌ 

![image](https://github.com/parvvaresh/sentiment-analysis-from-scratch-python-for-persian-tweets/blob/main/result_model.png)

----
follow me on : 

[![Twitter Badge](https://img.shields.io/badge/-Twitter-1da1f2?style=flat-square&labelColor=1da1f2&logo=twitter&logoColor=white&link=https://twitter.com/Yaronzz)](https://twitter.com/parvvaresh)
[![Email Badge](https://img.shields.io/badge/-Email-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:yaronhuang@foxmail.com)](mailto:parvvaresh@gmail.com)
[![Instagram Badge](https://img.shields.io/badge/-Instagram-purple?style=flat&logo=instagram&logoColor=white&link=https://instagram.com/parvvaresh/)](https://space.bilibili.com/7708412)
[![Github Badge](https://img.shields.io/badge/-Github-232323?style=flat-square&logo=Github&logoColor=white&link=https://space.bilibili.com/7708412)](https://github.com/parvvaresh)
