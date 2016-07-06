# FacebookChecksIn

Project develop to participate at kaggle competition proposed by facebook:

https://www.kaggle.com/c/facebook-v-predicting-check-ins


Data and details about the competition can be found on the website above.

# Solution proposed

## Model

Here, the use of a K-nearest neighbors classifier permit to quickly obtain good results. 

Because of the huge size of the dataset, it was cut on different grids based on the position of events (space-grids following x and y).

The classifier training was done on one grid extended with its nearest neighbors (to avoid side effect). Events close to midnight (time position), was duplicated before the begining of the day or after its end following its initial position, in order to avoid side effect for time.

## Result

The solution tested with cross validation in several cells gave a rate of 0.63.

Because of an unknown error, the final rate obtained on kaggle was around 0.00003 which is equivalent to a random guess: the submitted file was not correct.

Because of the huge size of the dataset and the two days it takes to compute the solution (with knn, the quickest algorithm!), the implementation mistake was not tracked and a better solution was not obtained.

The developped ideas were used to quickly extend an already existing script in kaggle's python 3 environment in order to make a submission. The rate obtained was 0.52883.


# File details

This work should be understand as a mining/understanding/solving work on the database. The jupyter notebooks represent the different steps of the research.


## Notebooks

00_mining -> load, cut and explore data

01_grip -> import and work on one cell to test different algorithms and data organizations

02_travel -> compare and plot some results with different classifiers

03_accuracy -> explore the importance of the accuracy parameter (unsed before that)

04_extension -> implement and test the grid extension (over time and space) for training

05_computation -> automatize the grid extension (since previous result was good)

06_execution -> execute the total script on a given list of cell

07_assistance -> copy of the previous notebook with a time control

08_reconstruction -> built the final csv file for submission from all partial results (obtained from each cell)

## Script

'submission.py' is a copy of the final submission on kaggle.



## Modules

'treatment.py' gather functions to read database and compute some results.

'figure.py' gather function for plotting.

