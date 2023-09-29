# Sailing Through Time: A Titanic Data Odyssey

---

## Project Description

**Description:** Our project was centered around the analysis of the Titanic dataset, which we obtained from Kaggle.com. Our primary research objective was to investigate the determinants of passenger survival and explore the application of machine learning techniques for predictive modeling. Specifically, we aimed to train a logistic regression model to discern patterns in the data and predict the likelihood of passenger survival, thus contributing to a deeper understanding of the factors influencing survival outcomes.

---

## Directory
1. [Project Description](#Project-Description) 
2. [Setup](#Setup)
3. [Dependencies](#Dependencies)
4. [Input Files](#Input-Files)
5. [Tableau Visualizations and Jupyter Notebook Files](#Tableau-Visualizations-and-upyter-Notebook-Files)
6. [Exploratory Analysis Charts](#Exploratory-Analysis-Charts)
7. [Analysis Presentation files](#Analysis-Presentation-files)
8. [Data Prep Analysis and Building ML Model](#Data-Prep-Analysis-and-Building-ML-Model)
9. [Conclusion](#Conclusion)
10. [References](#References)

---

## Setup:
- Imported the data from Kaggle Dataset - Titanic Machine Learning from Disaster
- Conducted exploratory data analysis to gain insight into the different classifications that were provided in the dataset
- Used tableau to visualize age and gender distribution, age group and gender by socio economic class, fare distribution, survival by gender and non survival by both socio economic class and gender.
- Used the logistic regression model to predict which passengers will survive and which will not.
- We used the random over sampler to resample the data.
- Tested the predictive power of socioeconomic status and the predictive power of gender.

---

## Dependencies

   import numpy as np
   
   import pandas as pd
   
   from pathlib import Path
   
   from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

---

## Input Files

- Training set (train.csv) - dataset representing outcome for each passenger.

---

## Tableau Visualizations and Jupyter Notebook Files
- titanic data - tableau visualizations
- titanic_survival_classification.ipynb
- titanic_survival_classification_no_pclass.ipynb
- titanic_survival_classification_no_null.ipynb
  
---

## Exploratory Analysis Charts

- Age Group and Gender Distribution.png
- Age Group and By Socio-Economic Class.png
- Gender by Socio-Economic Class.png
- Family Onboard.png
- Fare Distribution.png
- Departure City.png
- Passenger Class Distribution.png
- Did Not Survive by Fare.png
- Survival by Gender.png
- Did Not Survive by Socio-Economic Class.png
- Did Not Survive by Age Group and Gender.png
  
---

## Analysis Presentation files

- [PPT](./Project%204/Powerpoint.pptx)
- [Tableau visualizations link](https://public.tableau.com/app/profile/emily.rusin/viz/Project_4_16953464445020/GenderbySocio-EconomicClass?publish=yes)
  
---

## Data Prep Analysis and Building ML Model

- Loaded the CSV file  located on kaggle.com
- Split the Data into Training and Testing Sets
- Step 1: Read the data from the Resources folder into a Pandas DataFrame
- Step 2: Create the labels set (y) from the “Survival” column, and then create the features (X) DataFrame from the remaining columns
- Step 3: Check the balance of the labels variable (y) by using the value_counts function
- Step 4: Split the data into training and testing datasets by using train_test_split
- Create a Logistic Regression Model with the Original Data
- Step 1: Fit a logistic regression model by using the training data (X_train and y_train)
- Step 2: Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model
- Step 3: Evaluate the model’s performance by doing the following
- Predict a Logistic Regression Model with Resampled Training Data
- Step 1: Use the RandomOverSampler module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points
- Step 2: Use the LogisticRegression classifier and the resampled data to fit the model and make predictions
- Step 3: Evaluate the model’s performance by doing the following

---

  ## Insights

  **Age group and gender distribution**
  ![Age Group and Gender Distribution](Project%204/Exploratory%20analysis%20charts/Age%20Group%20and%20Gender%20Distribution.png)

- 64% of the passengers were male, mainly in the adult age range, which is 18 to 65 years old.  There were only 11 people over 65 (all male) and 7% were children.  We did not know the age of about 20% of the passengers.

**Age Group and By Socio-Economic Class**
 ![Age Group and By Socio-Economic Class](Project%204/Exploratory%20analysis%20charts/Age%20Group%20and%20By%20Socio-Economic%20Class.png)

- The lower-class passengers made up for the largest group of people onboard the Titanic.  There were very few children and teenagers in the Upper Class, with almost all the Senior passengers existing in the Upper Class.

**Gender by Socio-Economic Class**
![Gender by Socio-Economic Class](Project%204/Exploratory%20analysis%20charts/Gender%20by%20Socio-Economic%20Class.png)

- There were 577 males onboard and 314 females.  60% of the males were in the Lower Class while 46% of the females were categorized as lower class.  The females were more balanced in their status compared to the males.

**Family Onboard**
![Family Onboard](Project%204/Exploratory%20analysis%20charts/Family%20Onboard.png)

- About 40% of the passengers were travelling with family.  Our data was broken into a group of Sibling/Spouse and then a group with passengers travelling with a Parent/Child(ren).  We grouped these two categories into a “Family” category.   Fiancés, Mistresses, or nannies did not count towards this “family” category.

**Fare Distribution**

![Fare Distribution](Project%204/Exploratory%20analysis%20charts/Fare%20Distribution.png)

- 347 people paid between $5 and $10. 24% of Titanic Passengers paid approximately $7 for their fare, which is worth $221 today.The highest fare, $512, is worth over $16,000 today.

**Departure City**
![Departure City](Project%204/Exploratory%20analysis%20charts/Departure%20City.png)

- The Departure cities were Southampton, England, Cherbourg, France, and Queenstown, Ireland, which is Cobh, Ireland today.  72% of the passengers boarded in Southampton, England, which was the first port of the maiden voyage.

**Passenger Class Distribution**
![Passenger Class Distribution](Project%204/Exploratory%20analysis%20charts/Passenger%20Class%20Distribution.png)

- More Upper Class survived than perished, and more than 3x as many Lower Class passengers died than they did survive.  It was about equal for Middle Class, though a few more died than survived. 

**Did Not Survive by Fare**
![Did Not Survive by Fare](Project%204/Exploratory%20analysis%20charts/Did%20Not%20Survive%20by%20Fare.png)

- Over half (52%) of those who died in the sinking paid $10 or under.  Almost all of those who were riding for free died, and then the largest ticket group – those who paid $5 to $10 – 78% of those passengers died.  Once we hit about $71 per ticket ($2,285 today), more survived than died.

**Survival by Gender**
![Survival by Gender](Project%204/Exploratory%20analysis%20charts/Survival%20by%20Gender.png)

- 81% of the males and 1/4  of the females died.

**Did Not Survive by Socio-Economic Class**
![Did Not Survive by Socio-Economic Class](Project%204/Exploratory%20analysis%20charts/Did%20Not%20Survive%20by%20Socio-Economic%20Class.png)

- When looking at the total breakdown of people who died aboard the Titanic, almost 70% were lower class.  This would be expected.  It’s possible that the lower-class passengers had lower cabins and therefore were not granted initial seats on lifeboats.  More than likely, the wealthy were given precedent and their lives were perceived as more valuable.

**Did Not Survive by Age Group and Gender**
![Did Not Survive by Age Group and Gender](Project%204/Exploratory%20analysis%20charts/Did%20Not%20Survive%20by%20Age%20Group%20and%20Gender.png)

- The adult age group was the largest and had the largest number of deaths.  Almost all the passengers over 65 died (only 1 survived).  The youngest person to die was only 1 year old and the oldest passenger was 74.

---

## Conclusion

In our exploration of the predictive efficacy of gender and socio-economic status, our analysis reveals a noteworthy finding. The results indicate that gender emerged as the more influential factor in our model's predictive capabilities for survival rates, overshadowing the impact of socio-economic status.

---

## References

- Data for the csv file "Training set(train.csv)" was imported from https://www.kaggle.com
- Used Module 20 challenge as a template for the logistic regression model


