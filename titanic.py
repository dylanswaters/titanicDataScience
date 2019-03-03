import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import csv
import numpy

#ANOVA's
#read the train.csv file
trainDataCSV = pd.read_csv('titanicTrainingData.csv')

#create the sex ANOVA
aovModel = ols('Sex ~ Survived', data=trainDataCSV).fit()
# aov_table = sm.stats.anova_lm(aovModel, typ=2)
print("Sex Anova Results:")
# print(aov_table)

#create the passenger class ANOVA
aovModel = ols('Pclass ~ Survived', data=trainDataCSV).fit()
aov_table = sm.stats.anova_lm(aovModel, typ=2)
print("Passenger Class Anova Results:")
print(aov_table)

#male and female to survived lists
maleSurvivedList = []
maleSexList = []
femaleSurvivedList = []
femaleSexList = []

#multivariate lists
survivedList = []
pclassList = [1, 2, 3]
survivedByPClass = [0, 0 , 0]

ageList = []
fareList = []
survivedListAgeMissing = []

#read the train.csv file to get specific information
with open('train.csv') as datafile:
    output = csv.reader(datafile, delimiter=",")
    next(output)

    for row in output:
        #male and female to survived data fetching
        if(row[4] == 'male'):
            maleSurvivedList.append(float(row[1]))
            maleSexList.append(1)
        if(row[4] == 'female'):
            femaleSurvivedList.append(float(row[1]))
            femaleSexList.append(1)
        survivedList.append(float(row[1]))
        # pclassList.append(float(row[2]))
        fareList.append(float(row[9]))
        if(row[5] != ""):
            ageList.append(float(row[5]))
            survivedListAgeMissing.append(float(row[1]))
        #pclass survival count
        if(row[2] == '1'):
            survivedByPClass[0] += 1
        if(row[2] == '2'):
            survivedByPClass[1] += 1
        if(row[2] == '3'):
            survivedByPClass[2] += 1


# print("maleSexList")
# print(maleSexList)
# print("")
#
# print("maleSurvivedList")
# print(maleSurvivedList)
# print("")
#
# print("femaleSexList")
# print(femaleSexList)
# print("")
#
# print("femaleSurvivedList")
# print(femaleSurvivedList)
# print("")


#what are we supposed to do here?
linRegMale = numpy.polyfit(maleSurvivedList, maleSexList, 1)
linRegMale_fn = numpy.poly1d(linRegMale)
plt.figure(1)
plt.plot(maleSurvivedList, maleSexList, 'yo', maleSurvivedList, linRegMale_fn(maleSurvivedList), '--k')
plt.suptitle("Correlation between male and surviving")
plt.ylabel("Sex")
plt.xlabel("Survival")
# plt.show()
print("")

linRegFemale = numpy.polyfit(femaleSexList, femaleSurvivedList, 1)
linRegFemale_fn = numpy.poly1d(linRegFemale)
plt.figure(2)
plt.plot(femaleSexList, femaleSurvivedList, 'yo', femaleSexList, linRegFemale_fn(femaleSexList), '--k')
plt.suptitle("Correlation between female and surviving")
plt.ylabel("Survival")
plt.xlabel("Sex")
# plt.show()
print("")

#bivariate scatterplots
plt.figure(3)
plt.plot(pclassList, survivedByPClass, 'yo')
plt.suptitle("Count of survivors by passenger class")
plt.ylabel("Survivors")
plt.xlabel("Passenger class")
# plt.show()
print("")

#age vs Survival
plt.figure(4)
plt.plot(ageList, survivedListAgeMissing, 'yo')
plt.suptitle("age vs survival")
plt.ylabel("Survival")
plt.xlabel("Age")
# plt.show()

plt.figure(4)
plt.plot(fareList, survivedList, 'yo')
plt.suptitle("fare vs survival")
plt.ylabel("Survival")
plt.xlabel("Fare")
plt.show()
