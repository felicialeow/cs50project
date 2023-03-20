## What is it? 

This is a web application designed to conduct data exploration. It involves summary statistics and visualization methods. The project is part of the CS50 Introduction to Computer Science final assessment. 

## Installation 

1. Get a local copy of the project repo
```sh
git clone https://github.com/felicialeow/cs50project.git
```
2. Install packages 
```sh
pip install -r requirements.txt
```
3. Run the web application 
```sh
python -m flask run 
```

## Features 

To begin using the data exploratory tool, upload a CSV file that contains at least 2 rows of data and 2 columns of variables. 
There are 5 main features in the tool, it is recommended to explore each feature in order. 

### 1. Data Structure 

The first feature of the tool provides a snippet of the data set. There is an option to convert variable type from numerical to categorical or categorical to numerical (if appropriate). The variable type will affect the summary statistics and visualization method used subsequently. 
_It is possible to return to change the variable type._

### 2. Univariate Summary Statistics 

The first step into data exploration is univariate non-graphical analysis. Using several summary statistics it helps to provide a brief idea of the central tendency and spread of numerical data as well as frequency of category in categorical data. 

Below is a list of summary statistics available in our tool: 
* mean 
* standard deviation 
* quartiles 
* minimum and maximum 
* number of unique categories 
* number of missing value 

### 3. Univariate Visualizations

Apart from non-graphical method, univariate visualizations provide a more complete description of the data. It shows the shape of distribution, central tendency and spread of data. 

Numerical data are presented in **box plot** or **density plot** while categorical data are presented in **bar plot** and **pie chart**. 

### 4. Data Transformation 

Data transformation is a key process after univariate analysis. Based on the findings in univariate exploration, the raw data is altered in preparation for future analyses and application of statistical modelling.

There are several classes of transformation available: 
* mathematical operations 
* treatment of outlier 
* treatment of missing value 
* discretizing numerical variable by binning 
* regrouping/ renaming category label 

It can be useful to combine several transformation method to achieve desire outcome. 

### 5. Multivariate Visualizations 

The final feature of the tool is multivariate visualizations. It explores the relationship between two variables. It can compare the distribution of numerical data across different categories, understand the trend and pattern of numerical data, compute correlations among numerical data. Adding a third variable in the graphic, if applicable, shows the interaction with the other variables. 

There are 6 main types of visualization method supported:
1. scatter plot 
2. line plot 
3. box plot 
4. bar plot 
5. count plot 
6. density plot 

_It is not advisable to conduct multivariate analysis without exploring the individual data._

## Authors  
The project is created and maintained by [Felicia](https://github.com/felicialeow) and [Tian](https://github.com/GTdllab).  
Codespace taken from [Microsoft VS Code](https://github.com/microsoft/vscode-remote-try-python)