# Data Exploration Tool 

## What is it? 

This is a web application designed to conduct quick data exploration. It involves summary statistics and visualization methods. The project is part of the CS50 Introduction to Computer Science final assessment.  

A video has been created to demonstrate the usage of data exploration tool. <URL HERE>


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

To begin using the data exploratory tool, upload a CSV file that contains at least 2 rows of data and 2 columns of variables. _Any file that fails the criteria will not be able to proceed with the next steps._  
  
At any point, click the <span style="color: red"> reset </span> button on the top right corner of the page to remove all changes and re-upload data file.  

There are **5** main features in the tool, it is recommended to explore each feature in order.  

### 1. Data Structure 
<p style="font-size: 0.9rem">Relevant HTML file: [datatype.html](templates/datatype.html)</p>

The first feature of the tool provides a snippet of the data set. The first 5 rows of the data set is presented in a table. There is an option to convert variable type from numerical to categorical or categorical to numerical (if appropriate). The variable type will affect the summary statistics and visualization method used subsequently. 
_It is possible to return to change the variable type._  

Click the menu button on the top right corner to go to second feature - **Summary Statistics**

### 2. Univariate Summary Statistics 
<p style="font-size: 0.9rem">Relevant HTML files: [numeric.html](templates/numeric.html) and [categorical.html](templates/categorical.html)</p>

The first step into data exploration is univariate non-graphical analysis. Using several summary statistics it helps to provide a brief idea of the central tendency and spread of numerical data as well as frequency of category in categorical data. 

Below is the list of summary statistics available: 
- Numerical variable
    * mean 
    * standard deviation 
    * quartiles (25%, 50%, 75%)
    * minimum and maximum 
    * number of missing value
- Categorical variable
    * number of unique categories 
    * top frequency
    * number of missing value

Click the link in the page to toggle between numerical variables and categorical variables. 

There is an option to exclude any variable that may not be of interest, e.g. respondent id. _This exclusion is not permanent, it is possible to return to revert exclusions made._  

Click the menu button on the top right corner to go to third feature - **Univariate Visualizations**

### 3. Univariate Visualizations
<p style="font-size: 0.9rem">Relevant HTML files: [selectvariable.html](templates/selectvariable.html) and [univariateplot.html](templates/univariateplot.html)</p>

Univariate visualizations provide a more complete description of the data compared to non-graphical method. It shows the shape of distribution, central tendency and spread of data. 

Numerical data are presented in **box plot** (default) or **density plot** while categorical data are presented in **bar plot** (default) and **pie chart**. 

Key things to note: 
* For numerical data, it is important to identify skewed distribution and outliers. These may pose issues for future analyses 
* For categorical data, categories with low frequency count should be taken into consideration. It is less meaningful to analyze small group. Anomaly labels resulted from typing error should be identified and handled 

Click the menu button on the top right corner to go to forth feature - **Data Transformation**

### 4. Data Transformation 
<p style="font-size: 0.9rem">Relevant HTML files: [datatransform.html](templates/datatransform.html) and [datatransformed.html](templates/datatransformed.html)</p>

Data transformation is a key process after univariate analysis. Based on the findings in univariate exploration, the raw data is altered in preparation for future analyses and application of statistical modelling.

There are several classes of transformation available: 
* mathematical operations 
    * log transformation
    * exponential transformation
    * min-max scaling
    * normalization
* treatment of outlier 
    * replace with NaN
* treatment of missing value 
    * replace with specific value 
    * delete row
* discretizing numerical variable by binning 
* regrouping/ renaming category label 

It can be useful to combine several transformation method to achieve desire outcome. 

Click the menu button on the top right corner to go to last feature - **Multivariate Visualizations**

### 5. Multivariate Visualizations 
<p style="font-size: 0.9rem">Relevant HTML files: [selectmultivar.html](templates/selectmultivar.html) and [multivariateplot.html](templates/multivariateplot.html)</p>

The final feature of the tool is multivariate visualizations. It explores the relationship between two variables. It can compare the distribution of numerical data across different categories, understand the trend and pattern of numerical data, compute correlations among numerical data. Adding a third variable in the graphic, if applicable, shows the interaction with the other variables. 

There are 6 main types of visualization method supported:
1. scatter plot 
2. line plot 
3. box plot 
4. bar plot 
5. count plot 
6. density plot 

There are several options to alter the graphic view: expand x axis, expand y axis, switch x and y axis and sort category label (if applicable)

_It is not advisable to conduct multivariate analysis without exploring the individual data._

## Authors  
The project is created and maintained by [Felicia](https://github.com/felicialeow) and [Tian](https://github.com/GTdllab).  
Codespace taken from [Microsoft VS Code](https://github.com/microsoft/vscode-remote-try-python)