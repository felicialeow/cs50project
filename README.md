# install requirements

pip install -r requirements.txt

# web tool workflow 

1. main landing page 
- description of the web app 
- control to upload raw data file
    - check if data file has at least 2 rows and 2 columns 
Note: included a sample.csv file - which is from the adult.csv file shared on Skype

https://thinkinfi.com/flask-upload-display-file/


2. data structure 
- show number of rows and columns
- show the first 5 rows in the dataset 
- decide if the variable is numeric (integer) or categorical (string)
    - if the data type is not changed, the numeric variable is either integer or float
        depending on how pandas interpreted it 

3. table of summary statistics 
this is split into numeric variables and categorical variables 
- numeric variables:
    - count
    - mean
    - std
    - min
    - 25%, 50% and 75% quartile 
    - max
- categorical variables:
    - count 
    - number of unique value
    - category with top count
    - top count
- option to remove variable from the dataset

4. univariate visualisation 
- select numeric or categorical variable 
- select variable 
- plot type for numeric variable 
    - box plot [default]
    - density plot (density line based on gaussian estimation)
- plot type for categorical variable
    - bar plot [default]
    - pie chart 

5. feature engineering 

# https://www.kaggle.com/code/alokevil/simple-eda-for-beginners
chart builder page 
- show data structure
- select x variable 
    - show various measure calculations
- select y variable
- check if x and y are different [Felicia stopped here]
- show preview of summarized data 
- select filter variable 
    - show sample size of each group after selection in a table format
- select chart type 
    - show image of different type of chart 

    https://pypi.org/project/Flask-Plots/
    https://matplotlib.org/stable/gallery/user_interfaces/web_application_server_sgskip.html

chart design page 
- show chart with default design 
- depending on the type of chart, option to change setting 
    1. line chart 
        - line color 
        - line type
        - line thickness
    2. scatter plot 
        - point shape 
        - point color 
    3. bar chart 
        - bar color 
    4. pie chart 
        - pie color