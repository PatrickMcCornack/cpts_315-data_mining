# -*- coding: utf-8 -*-
"""
PA 3 
Author: Pat McCornack
Adapted from code provided in Panopto video. 

11/13/2022
"""

import pandas as pd
import numpy as np
import nltk
import re
# import re
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


## stop words
stop = stopwords.words('english')


## Read Files
x = pd.read_csv("./fortune-cookie-data/traindata.txt")
y = pd.read_csv("./fortune-cookie-data/trainlabels.txt")
z = pd.read_csv("./fortune-cookie-data/stoplist.txt")
predData = pd.read_csv("./fortune-cookie-data/testdata.txt")
predLabels = pd.read_csv("./fortune-cookie-data/testlabels.txt")

## Write file
new_file = 'Binary Classifer \n'

## Train data size
rangeX = x.size

## Test Data size
rangePred = predData.size

## Rename columsn
x.columns = ['Data']
y.columns = ['Label']
z.columns = ['Stop']
predData.columns = ['Data']
predLabels.columns = ['Label']

## Combine train and test
dx = pd.concat([x, predData])

## List of stop words
stopwords = z["Stop"].tolist()

## Create tokenized data
# Apply lambda function to remove words in stopwords list 
dx['Filt_data'] = dx['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
del dx['Data']

# Take each filtered string and convert to list
dx['Tokenized_Data'] = dx.apply(lambda row: nltk.word_tokenize(row['Filt_data']), axis=1)

## Train labels to list
y = y['Label'].tolist()

## Test labels to list
predLabels = predLabels['Label'].tolist()

## Training Data - TFIDF
v = TfidfVectorizer()

Tfidf = v.fit_transform(dx['Filt_data'])

df1 = pd.DataFrame(Tfidf.toarray(), columns = v.get_feature_names_out())


## Separating train and test
x = df1[0:rangeX]
predData = df1[rangeX:rangeX+rangePred]

### Perceptron implementation using scikit-learn
## Perceptron implementation
# ppn = Perceptron(max_iter = 20, eta0 = 1, random_state = 0, verbose = 1)

# Fitting x, y
# ppn.fit(x,y)

# Using Test data
# y_pred = ppn.predict(predData)

# Accuracy Calculation
# print('Accuracy: %.2f' % accuracy_score(predLabels, y_pred))


### Normal perceptron manual implementation
w = [0] * x.shape[1]
iterations = 20
learn_rate = 1
y_hat_list = []
num_mistakes = [0] * iterations
training_accuracy = [0] * iterations
testing_accuracy= [0] * iterations
weights = []

print("Fortune Cookie Classifier: ")

for itr in range(iterations):
    
    ## Calculate the weight vector
    for i in range(x.shape[0]):
        x_t = x.iloc[i].tolist()
        y_hat = np.sign(np.dot(x_t, w))
        if y_hat == -1:
            y_hat = 0  # The training labels are 0 or 1
        
        if y_hat != y[i]:
            update = 1
            if y[i] == 0:
                update = -1
            
            num_mistakes[itr] += 1
            w = np.add(w, np.dot((learn_rate * update), x_t)).tolist()
        
     
     
    ## Check the training accuracy
    y_hat_list.clear()  # Empty list to check accuracy of this iteration
    
    # Make predictions
    for i in range(x.shape[0]):
        x_t = x.iloc[i].tolist()
        y_hat = np.sign(np.dot(x_t,w))
        if y_hat == -1:
            y_hat_list.append(0)
        else:
            y_hat_list.append(1)
        
    # Compare predictions to true values
    match = 0
    for i in range(len(y_hat_list)):
        if(y_hat_list[i] == y[i]):
            match += 1
    training_accuracy[itr] = match / x.shape[0]
    
    
    ## Check the testing results
    y_hat_list.clear()
    
    # Make predictions
    for i in range(predData.shape[0]):
        pred_t = predData.iloc[i].tolist()
        y_hat = np.sign(np.dot(pred_t,w))
        if y_hat == -1:
            y_hat_list.append(0)
        else:
            y_hat_list.append(1)
        
    # Compare predictions to true values
    match = 0
    for i in range(len(y_hat_list)):
        if(y_hat_list[i] == predLabels[i]):
            match += 1
    testing_accuracy[itr] = match / predData.shape[0]
    
training_accuracy = [round(item, 3) for item in training_accuracy]
testing_accuracy = [round(item, 3) for item in testing_accuracy]


print("a. \nThe number of mistakes the standard perceptron made during each iteration is: " + str(num_mistakes))
print("\nb. \nThe training accuracy of the standard perceptron after each iteration is: " + str(training_accuracy))
print("\nThe testing accuracy of the standard perceptron after each iteration is: " + str(testing_accuracy))

print("\nc. \nThe training accuracy of the standard perceptron after 20 iterations is: " + str(round(training_accuracy[19], 2)))
print("The testing accuracy of the standard perceptron after 20 iterations is: " + str(round(testing_accuracy[19], 2)))

# Write number of mistakes
for item in num_mistakes:
    new_file = new_file + str(item) + '\n'

new_file = new_file + '\n'

# Write training/testing accuracy per iteration
for i in range(len(training_accuracy)):
    new_file = new_file + str(training_accuracy[i]) + " " + str(testing_accuracy[i]) + '\n'

new_file = new_file + '\n'

std_accuracy = [training_accuracy[19], testing_accuracy[19]]  # Used to write to file later


### Averaged Perceptron

print("\n Implementing Averaged Perceptron... ")
w = [0] * x.shape[1]
u = [0] * x.shape[1]
c = 1
iterations = 20
learn_rate = 1
y_hat_list = []
num_mistakes = [0] * iterations
training_accuracy = [0] * iterations
testing_accuracy= [0] * iterations
weights = []

for itr in range(iterations):
    
    ## Calculate the weight vector
   
    for i in range(x.shape[0]):
        x_t = x.iloc[i].tolist()
        y_hat = np.sign(np.dot(x_t, w))
        if y_hat == -1:
            y_hat = 0  # The training labels are 0 or 1
        
        if y_hat != y[i]:
            update = 1
            if y[i] == 0:
                update = -1
            
            num_mistakes[itr] += 1
            w = np.add(w, np.dot((learn_rate * update), x_t)).tolist()
            u = np.add(u, np.dot((learn_rate * update * c), x_t)).tolist()
        c += 1
     
     
    ## Check the training accuracy
    y_hat_list.clear()  # Empty list to check accuracy of this iteration
    
    # Make predictions
    for i in range(x.shape[0]):
        x_t = x.iloc[i].tolist()
        y_hat = np.sign(np.dot(x_t, np.subtract(w, np.divide(u,c).tolist()).tolist()))
        if y_hat == -1:
            y_hat_list.append(0)
        else:
            y_hat_list.append(1)
        
    # Compare predictions to true values
    match = 0
    for i in range(len(y_hat_list)):
        if(y_hat_list[i] == y[i]):
            match += 1
    training_accuracy[itr] = match / x.shape[0]
    
    
    ## Check the testing results
    y_hat_list.clear()
    
    # Make predictions
    for i in range(predData.shape[0]):
        pred_t = predData.iloc[i].tolist()
        y_hat = np.sign(np.dot(pred_t,w))
        if y_hat == -1:
            y_hat_list.append(0)
        else:
            y_hat_list.append(1)
        
    # Compare predictions to true values
    match = 0
    for i in range(len(y_hat_list)):
        if(y_hat_list[i] == predLabels[i]):
            match += 1
    testing_accuracy[itr] = match / predData.shape[0]

print("\nThe training accuracy of the averaged perceptron after 20 iterations is: " + str(round(training_accuracy[19], 3)))
print("The testing accuracy of the averaged perceptron after 20 iterations is: " + str(testing_accuracy[19]))

new_file += str(std_accuracy[0]) + ' ' + str(std_accuracy[1]) + '\n'
new_file += str(training_accuracy[19]) + ' ' + str(testing_accuracy[19]) + '\n\n' # Averaged perceptron accuracies


    
# OCR
##################################################################
print("\n\n Implementing Optical Character Recognition... ")

# Create a List from String
def split(word):
    return list(word)

# Pre-process Data Files
def pre_process(file):
    
    print("Pre-processing data...")
    # Read all lines
    with open(file, 'r') as f:
        lines = f.readlines()
        
    # Preprocessing Steps
    # Filter Empty Lines
    # Separate Index, Data, Label
    # Preprocess Data
    # Create Labels
    
    i = 0
    for line in lines:
        l = re.split(r'\t+', line)
        
        if len(l) > 2:
            
            # print (l)
            # print(l[0])
            l1 = split(l[1][2:len(l[1])])
            dsl = len(l1)
            # print("data string length: ", len(l1))
            if i == 0:
                k1 = np.array(l1)
                k = k1
                label1 = np.array(l[2])
                label = label1
            else: 
                k1 = np.array(l1)
                k = np.append(k, k1)
                label1 = np.array(l[2])
                label = np.append(label, label1)
            # print(y)
            i = i + 1
            
        # print(i)
        
    # Reshape
    k2 = k.reshape(i, dsl)
    
    # Data Frame from Array
    dataframe = pd.DataFrame.from_records(k2)
    return(dataframe, label)


# Implement Perceptron
def implement_perceptron(train_data, train_labels, test_data, test_labels):
    
    w = []
    iterations = 20
    learn_rate = 1
    labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    k = 26
    train_data = train_data.astype('int')
    test_data = test_data.astype('int')
    pred_list = []
    
    num_mistakes = [0] * iterations
    training_accuracy = [0] * iterations
    testing_accuracy= [0] * iterations
    
    
    ## Calculate the weight vector
    print("Calculating weight vector... ")
    
    #initialize the training weights
    for i in range(k):
        w.append(([0] * train_data.shape[1]))
                             
    for itr in range(iterations):
        for i in range(train_data.shape[0]):
            x_t = train_data.iloc[i].tolist()
            
            # Find weight vector that maximizes the expression w * x
            maximum = [np.dot(x_t, w[0]), 0]
            pred_label = labels[0]
            for j in range(k):
                dot = np.dot(x_t, w[j])
                if dot > maximum[0]:
                    maximum = [dot, j]
                    pred_label = labels[j]
            
            # If the predicted label is wrong
            if pred_label != train_labels[i]:
                num_mistakes[itr] += 1
                w_index = labels.index(train_labels[i])  # index of weight vector corresponding to label
                w[w_index] = w[w_index] + np.dot(learn_rate, x_t)  # Updates weight of actual label
                w[maximum[1]] = w[maximum[1]] - np.dot(learn_rate, x_t) # Updates weight of predicted label 
    
    
    ## Check the accuracy
    
    ## Check the training accuracy
        print("Checking training accuracy...")
        pred_list.clear()  # Empty list to check accuracy of this iteration
         
        # Make predictions
        for i in range(train_data.shape[0]):
            x_t = train_data.iloc[i].tolist()
            
            # Find weight vector that maximizes the expression w * x
            maximum = [np.dot(x_t, w[0]), 0]
            for j in range(k):
                dot = np.dot(x_t, w[j])
                if dot > maximum[0]:
                    maximum = [dot, j]
                    pred_label = labels[j]
            
            pred_list.append(pred_label)
    
        
        # Compare predictions to true values
        match = 0
        for i in range(len(pred_list)):
            if(pred_list[i] == train_labels[i]):
                match += 1
        training_accuracy[itr] = match / train_data.shape[0]
    
    
    ## Check the testing accuracy
        print("Checking testing accuracy... ")
        pred_list.clear()  # Empty list to check accuracy of this iteration
        
        # Make predictions
        for i in range(test_data.shape[0]):
            x_t = test_data.iloc[i].tolist()
            
            # Find weight vector that maximizes the expression w * x
            maximum = [np.dot(x_t, w[0]), 0]
            for j in range(k):
                dot = np.dot(x_t, w[j])
                if dot > maximum[0]:
                    maximum = [dot, j]
                    pred_label = labels[j]
            
            pred_list.append(pred_label)
    
        
        # Compare predictions to true values
        match = 0
        for i in range(len(pred_list)):
            if(pred_list[i] == test_labels[i]):
                match += 1
        testing_accuracy[itr] = match / test_data.shape[0]
        
    print("\na. \nThe number of mistakes the OCR perceptron made during each iteration is: " + str(num_mistakes))
    print("\nb. \nThe training accuracy of the OCR perceptron after each iteration is: " + str(training_accuracy))
    print("\nThe testing accuracy of the OCR perceptron after each iteration is: " + str(testing_accuracy))
    
    print("\nc. \nThe training accuracy of the OCR perceptron after 20 iterations is: " + str(round(training_accuracy[19], 4)))
    print("The testing accuracy of the OCR perceptron after 20 iterations is: " + str(round(testing_accuracy[19], 4)))
    
    return num_mistakes, training_accuracy, testing_accuracy
    
    
# Read train data
train = r"./OCR-data/OCR-data/ocr_test.txt"

# Read test data
test = r"./OCR-data/OCR-data/ocr_train.txt"

# Pre-process Train Data
train_data, train_labels = pre_process(train)

# Pre-process Test Data
test_data, test_labels = pre_process(test)

# Implement Perceptron
num_mistakes, training_accuracy, testing_accuracy = implement_perceptron(train_data, train_labels, test_data, test_labels)
new_file = new_file + "OCR: " + '\n'

# Write number of mistakes
for item in num_mistakes:
    new_file = new_file + str(item) + '\n'

new_file = new_file + '\n'

# Write training/testing accuracy per iteration
for i in range(len(training_accuracy)):
    new_file = new_file + str(training_accuracy[i]) + " " + str(testing_accuracy[i]) + '\n'

new_file = new_file + '\n'

std_accuracy = [training_accuracy[19], testing_accuracy[19]]  # Used to write to file later
new_file = new_file + str(training_accuracy[19]) + ' ' + str(testing_accuracy[19])

with open('output.txt', "w+") as f:  # Creates a new csv file
    f.write(new_file)





