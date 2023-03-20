# Required libraries
from flask import Flask, render_template, request, session, redirect, render_template_string
from flask_socketio import SocketIO
import pandas as pd
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import math
import string


# Flask configuration
app = Flask(__name__)
upload_folder = os.path.join('static', 'uploadfile')
app.config['UPLOAD_FOLDER'] = upload_folder
app.secret_key = "secret"


# Helper function 1: clear directory
def clear_dir():
    filelist = [file for file in os.listdir(app.config['UPLOAD_FOLDER'])]
    for file in filelist:
        os.remove(os.path.join(upload_folder, file))


# Helper function 2: start session
def start_session():
    list_of_info = ['uploaded_data_file_path',
                    'uploaded_data_backup_filepath',
                    'vartype_filepath',
                    'vartype_FIXED_filepath',
                    'excludedvar_filepath',
                    'fig_sizex',
                    'fig_sizey']
    for info in list_of_info:
        session[info] = ""
    session['fig_sizex'] = 7
    session['fig_sizey'] = 4


# Helper function 3: clear session information
def clear_session(list_of_info):
    for info in list_of_info:
        session[info] = ""


# Helper function 4: calculate percentage in pie chart
def calculate_percentage(count, allvals):
    percent = numpy.round(count/numpy.sum(allvals)*100)
    return f"{percent:.1f}%"


# Helper function 5: get color value for discrete groups
def get_color(n):
    cmap = plt.get_cmap('gist_rainbow', 100)
    color_values = cmap([val*int(100/(n-1)) for val in range(n)])
    return color_values


# Helper function 6: generate list of formula for data transformation
def generate_formula(vartype):
    all_formula = [
        ['Log-transformation', 'Taking natural logs will transform right-skewed data to make the distribution more symmetrical and closer to normal distribution', 'Numerical'],
        ['Exponential-transformation',
            'Taking exponential will multiple the effect of a unit change in value', 'Numerical'],
        ['Min-Max_Scaler', 'Data normalization such that all values are in the range of 0 and 1', 'Numerical'],
        ['Standardization', 'Rescaling the distribution of values so that the mean is 0 and standard deviation is 1', 'Numerical'],
        ['Binning', 'Transform numerical variable into categorical variable by grouping intervals of values into categories', 'Numerical'],
        ['Outlier-Anomaly_Treatment', 'Replacing outlier/anomaly with NaN', 'Numerical'],
        ['Renaming_Label',
            'Renaming categorical label (multiple labels can be replaced with the same label)', 'Categorical'],
        ['Replace_Missing_Value', 'Replacing NaN with specific value', 'Both'],
        ['Delete_Missing_Value', 'Remove row with missing value', 'Both']
    ]
    if vartype == 'both':
        return all_formula
    elif vartype == 'numeric':
        return [f for f in all_formula if (f[2] == 'Numerical') or (f[2] == 'Both')]
    else:
        return [f for f in all_formula if (f[2] == 'Categorical') or (f[2] == 'Both')]


# Helper function 7: write files
def write_files(df, vartype_df, excludedvar_df, new_row, newvar):
    new_row = pd.DataFrame(new_row)
    # check if row exist in variable type file
    if len(vartype_df.loc[vartype_df.column == newvar, :]) == 0:
        vartype_df = pd.concat([vartype_df, new_row], ignore_index=True)
    # check if row exist in excluded variable file
    if len(excludedvar_df.loc[excludedvar_df.column == newvar, :]) == 0:
        excludedvar_df = pd.concat([excludedvar_df, pd.DataFrame(
            {'column': [newvar], 'exclude': [False]})], ignore_index=True)
    df.to_csv(session.get(
        'uploaded_data_file_path'), index=False)
    vartype_df.to_csv(session.get(
        'vartype_filepath'), index=False)
    excludedvar_df.to_csv(session.get(
        'excludedvar_filepath'), index=False)


# Helper function 8: set parameters of all text in figure
def set_figure_params():
    smallfontsize = 5
    normalfontsize = 7
    largefontsize = 9
    params = {
        'axes.labelsize': normalfontsize,
        'axes.labelweight': 'bold',
        'font.size': normalfontsize,
        'legend.fontsize': smallfontsize,
        'xtick.labelsize': normalfontsize,
        'ytick.labelsize': normalfontsize,
        'figure.titlesize': largefontsize,
        'figure.titleweight': 'bold'
    }
    plt.rcParams.update(params)


set_figure_params


# Helper function 9: check if categorical variable can be converted to numeric
def testnumeric(var, df):
    def numericval(value):
        try:
            float(value)
            f = True
        except:
            f = False

        try:
            int(value)
            i = True
        except:
            i = False
        return f or i

    var_df = df[var]
    return all([numericval(value) for value in var_df])


# Helper function 10: generate list of plot
def generate_plottype(combi):
    allplots = [['scatter-plot', 'numeric-numeric', 'Shows the correlation between x-variable and y-variable. It describes how x-variable changes with y-variable.'],
                ['line-plot', 'numeric-numeric',
                 'Shows the linear correlation between x-variable and y-variable.'],
                ['box-plot', 'numeric-categorical',
                 'Compare the range and spread of x-variable across categories of y-variable.'],
                ['bar-plot_sum', 'numeric-categorical',
                 'Sum of x-variable for each group of y-variable'],
                ['bar-plot_average', 'numeric-categorical',
                 'Average of x-variable for each group of y-variable'],
                ['count-plot', 'categorical-categorical',
                 'Count of x-variable across each group of y-variable. NOTE: Limit to two groups of y-variable'],
                ['density-plot', 'numeric-categorical',
                 'Compare the distribution of x-variable for each group of y-variable']
                ]
    if combi == 'all':
        return allplots
    elif (combi == 'numeric-categorical') or (combi == 'categorical-numeric'):
        return [plot for plot in allplots if ('numeric-categorical' == plot[1]) or ('categorical-numeric' == plot[1])]
    else:
        return [plot for plot in allplots if combi in plot[1]]


# Upload file
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        clear_dir()
        start_session()
        return render_template('index.html', success="initial")
    else:
        # user uploaded file
        if request.files.get('uploaded-file'):
            # flask file
            uploaded_file = request.files['uploaded-file']
            # filename
            filename = secure_filename(uploaded_file.filename)
            # upload file to database
            uploaded_file.save(os.path.join(
                app.config['UPLOAD_FOLDER'], filename))
            # store file path in session
            session['uploaded_data_file_path'] = os.path.join(
                app.config['UPLOAD_FOLDER'], filename)
            # store backup file
            uploaded_file.save(os.path.join(
                app.config['UPLOAD_FOLDER'], 'backup.csv'))
            session['uploaded_data_backup_filepath'] = os.path.join(
                app.config['UPLOAD_FOLDER'], 'backup.csv')

            # check file format and data structure
            file_extension = os.path.splitext(filename)[1]
            error = 0
            if file_extension == '.csv':
                df = pd.read_csv(session.get('uploaded_data_file_path'))
                if (df.shape[0] >= 2 and df.shape[1] >= 2):
                    error = 0
                else:
                    error = 1
            else:
                error = 1

            # valid file
            if error == 0:
                # determine variable type
                colnames = df.columns.tolist()
                colclass = []
                coltype = []
                for col in colnames:
                    if isinstance(df[col][0], numpy.int64):
                        colclass.append('numeric')
                        coltype.append('int')
                    elif isinstance(df[col][0], numpy.float64):
                        colclass.append('numeric')
                        coltype.append('float')
                    else:
                        colclass.append('categorical')
                        coltype.append('str')
                # save variable type to database
                vartype = pd.DataFrame(
                    {'column': colnames, 'class': colclass, 'type': coltype})
                vartype.to_csv(os.path.join(
                    app.config['UPLOAD_FOLDER'], 'vartype.csv'), index=False)
                # store file path of variable type in session
                session['vartype_filepath'] = os.path.join(
                    app.config['UPLOAD_FOLDER'], 'vartype.csv')
                # create a backup copy of variable type file
                vartype_FIXED = vartype.copy()
                vartype_FIXED.to_csv(os.path.join(
                    app.config['UPLOAD_FOLDER'], 'vartype_FIXED.csv'), index=False)
                session['vartype_FIXED_filepath'] = os.path.join(
                    app.config['UPLOAD_FOLDER'], 'vartype_FIXED.csv')
                # create empty dataframe to store excluded variables
                excludedvar = pd.DataFrame(
                    {'column': colnames, 'exclude': False})
                excludedvar.to_csv(os.path.join(
                    app.config['UPLOAD_FOLDER'], 'excludedvar.csv'), index=False)
                # store file path of excluded variables in session
                session['excludedvar_filepath'] = os.path.join(
                    app.config['UPLOAD_FOLDER'], 'excludedvar.csv')
                return render_template('index.html', success="yes")
            # remove invalid data upload
            else:
                clear_dir()
                start_session()
                error_message = []
                if file_extension != '.csv':
                    error_message.append('Data file needs to be .csv format')
                else:
                    if df.shape[0] < 2:
                        error_message.append(
                            'Data needs to have at least 2 rows')
                    if df.shape[1] < 2:
                        error_message.append(
                            'Data needs to have at least 2 columns')
                return render_template('index.html', success="no", messages=error_message)


# Data structure
@app.route('/datatype', methods=['GET', 'POST'])
def datatype():
    # current session
    if request.method == 'POST':
        # file path
        file_path = session.get('uploaded_data_file_path')
        # read data file
        df = pd.read_csv(file_path)

        excludedvar_df = pd.read_csv(session.get('excludedvar_filepath'))

        df = df[excludedvar_df[excludedvar_df['exclude']
                               == False]['column'].tolist()]
        html_table = df.head().to_html(index=False, header=True, classes='table-style')
        colnames = df.columns.tolist()

        # read variable type file
        if request.form.get('reset') == 'datatype':
            vartype = pd.read_csv(session.get('vartype_FIXED_filepath'))

        else:
            vartype = pd.read_csv(session.get('vartype_filepath'))
        vartype = vartype[vartype['column'].isin(excludedvar_df[excludedvar_df['exclude']
                                                                == False]['column'].tolist())]
        # check if categorical variable can be converted into numerical variable
        vartype_list = list(zip(vartype['column'], vartype['class']))
        var_fixed = []
        for var in vartype_list:
            if var[1] == 'numeric':
                var_fixed.append(True)
            else:
                var_fixed.append(not (testnumeric(var[0], df)))
        vartype_list = list(
            zip(vartype['column'], vartype['class'], var_fixed))

        # user redirected to datatype.html
        if request.form.get('redirect') == 'datatype':
            return render_template('datatype.html', df=[html_table], shape=df.shape, vartype=vartype_list, submit='no')
        # user submitted the form to change variable type
        else:
            for col in colnames:
                newcolclass = request.form.get(col)
                # update variable type if it is different from current
                if newcolclass != vartype.loc[vartype['column'] == col, 'class'].values[0]:
                    vartype.loc[vartype['column'] ==
                                col, 'class'] = newcolclass
                    if newcolclass == 'numeric':
                        vartype.loc[vartype['column']
                                    == col, 'type'] = 'int'
                    else:
                        vartype.loc[vartype['column']
                                    == col, 'type'] = 'str'
            # save changes in variable type file
            vartype.to_csv(session.get(
                'vartype_filepath'), index=False)
            vartype_list = list(zip(vartype['column'], vartype['class']))
            return render_template('datatype.html', df=[html_table], shape=df.shape, vartype=vartype_list, submit='yes')
    # restart session
    else:
        return redirect('/')


# Description of numerical variables
@app.route('/numeric', methods=['GET', 'POST'])
def numeric():
    # current session
    if request.method == 'POST':
        # file path
        file_path = session.get('uploaded_data_file_path')
        # read data file
        df = pd.read_csv(file_path)
        # read variable type file
        vartype = pd.read_csv(session.get('vartype_filepath'))
        numericvar = vartype.loc[vartype['class']
                                 == 'numeric', 'column'].values.tolist()
        # read excluded variable file
        excludedvar_df = pd.read_csv(session.get('excludedvar_filepath'))

        # user submitted the form to exclude variable
        if request.form.get('exclude') == 'Exclude':
            # update excluded variables file
            col = request.form.get('remove-numericvar')
            excludedvar_df.loc[excludedvar_df.column == col, 'exclude'] = True
            excludedvar_df.to_csv(session.get(
                'excludedvar_filepath'), index=False)
        # user submitted the form to revert all exclusions
        elif request.form.get('reset') == 'Revert Exclusions':
            excludedvar_df.loc[excludedvar_df.column.isin(
                numericvar), 'exclude'] = False
            excludedvar_df.to_csv(session.get(
                'excludedvar_filepath'), index=False)
        excludedvar = excludedvar_df.loc[excludedvar_df.exclude,
                                         'column'].values.tolist()
        # subset dataset
        selectedvar = [var for var in numericvar if var not in excludedvar]

        # no numerical variables
        if len(selectedvar) == 0:
            return render_template('numeric.html', nvar=0)
        else:
            new_df = df[selectedvar].copy()
            # update variable type
            for col in selectedvar:
                if vartype.loc[vartype['column'] == col, 'type'].values[0] == 'int':
                    # if create Na value in data transformation, will have this error
                    # pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer
                    new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                    # create a boolean mask for not NA values
                    mask = new_df[col].notna()
                    # convert not NA values to integer
                    new_df.loc[mask, col] = new_df.loc[mask,
                                                       col].astype(numpy.int64)
                else:
                    new_df[col] = new_df[col].astype(numpy.float64)
            # summary statistics of variables
            df_desc = new_df.describe().round(2)
            df_desc = pd.concat(
                [df_desc, new_df.isnull().sum(axis=0).to_frame('NA').transpose()])
            html_table = df_desc.to_html(header=True, classes='table-style')
            nvar = df_desc.shape[1]
            return render_template('numeric.html', nvar=nvar, df=html_table, columns=selectedvar)
    # restart session
    else:
        return redirect('/')


# Description of categorical variables
@app.route('/categorical', methods=['GET', 'POST'])
def categorical():
    # current session
    if request.method == 'POST':
        # file path
        file_path = session.get('uploaded_data_file_path')
        # read data file
        df = pd.read_csv(file_path)
        # read variable type file
        vartype = pd.read_csv(session.get('vartype_filepath'))
        categoricalvar = vartype.loc[vartype['class']
                                     == 'categorical', 'column'].values.tolist()
        # read excluded variable file
        excludedvar_df = pd.read_csv(session.get('excludedvar_filepath'))

        # user submitted the form to exclude variable
        if request.form.get('exclude') == 'Exclude':
            # update excluded variables file
            col = request.form.get('remove-categoricalvar')
            excludedvar_df.loc[excludedvar_df.column == col, 'exclude'] = True
            excludedvar_df.to_csv(session.get(
                'excludedvar_filepath'), index=False)
        # user submitted the form to revert all exclusions
        elif request.form.get('reset') == 'Revert Exclusions':
            excludedvar_df.loc[excludedvar_df.column.isin(
                categoricalvar), 'exclude'] = False
            excludedvar_df.to_csv(session.get(
                'excludedvar_filepath'), index=False)
        excludedvar = excludedvar_df.loc[excludedvar_df.exclude,
                                         'column'].values.tolist()
        # subset dataset
        selectedvar = [var for var in categoricalvar if var not in excludedvar]

        # no categorical variables
        if len(selectedvar) == 0:
            return render_template('categorical.html', nvar=0)
        else:
            new_df = df[selectedvar].copy()

            na_row = new_df.isna().sum(axis=0).to_frame('NA').transpose()
            # update variable type
            for col in selectedvar:
                new_df[col] = new_df[col].astype(str)
            # summary statistics of variables
            df_desc = new_df.describe()
            str_length = [new_df[col].map(str).apply(
                len).max() for col in selectedvar]
            df_desc = pd.concat([df_desc, pd.DataFrame(
                {'max. length': str_length}, index=selectedvar).transpose()])

            df_desc = pd.concat(
                [df_desc, na_row])

            html_table = df_desc.to_html(header=True, classes='table-style')
            nvar = df_desc.shape[1]
            # categorical variable with length of label exceeding 20
            long_label = [selectedvar[i]
                          for i in range(nvar) if str_length[i] > 20]

            return render_template('categorical.html', nvar=nvar, df=html_table, columns=selectedvar, long_label=long_label)
    # restart session
    else:
        return redirect('/')


# Select variable for univariate visualization
@ app.route('/selectvariable', methods=['GET', 'POST'])
def selectvariable():
    # current session
    if request.method == 'POST':
        # read all files
        vartype_df = pd.read_csv(session.get('vartype_filepath'))
        excludedvar_df = pd.read_csv(session.get('excludedvar_filepath'))
        includedvar = excludedvar_df.loc[~excludedvar_df['exclude'], 'column'].tolist(
        )
        # categorical variables
        categoricalvar = [
            col for col in includedvar if col in vartype_df.loc[vartype_df['class'] == 'categorical', 'column'].values]
        # numerical variables
        numericvar = [
            col for col in includedvar if col in vartype_df.loc[vartype_df['class'] == 'numeric', 'column'].values]
        ncategorical = len(categoricalvar)
        nnumeric = len(numericvar)

        # user redirected to selectvariable.html
        if request.form.get('redirect') == 'selectvariable':
            types = []
            if ncategorical > 0:
                types.append('categorical')
            if nnumeric > 0:
                types.append('numeric')
            # only one variable type available
            if len(types) == 1:
                if types[0] == 'numeric':
                    return render_template('selectvariable.html', types=types, variables=numericvar, submit='no')
                else:
                    return render_template('selectvariable.html', types=types, variables=categoricalvar, submit='no')
            else:
                return render_template('selectvariable.html', types=types, submit='no')
        # user submitted the form to select variable type
        elif request.form.get('submit') == 'Submit':
            types = [request.form.get('selected-datatype')]
            # user hasnt select variable
            if not request.form.get('selected-var'):
                if types[0] == 'numeric':
                    return render_template('selectvariable.html', types=types, variables=numericvar, submit='no')
                else:
                    return render_template('selectvariable.html', types=types, variables=categoricalvar, submit='no')
            else:
                variables = [request.form.get('selected-var')]
                return render_template('selectvariable.html', types=types, variables=variables, submit='yes')
    # restart session
    else:
        return redirect('/')


# Univariate plot
@ app.route('/univariateplot', methods=['GET', 'POST'])
def univariateplot():
    # user redirected to univariateplot.html
    if request.method == 'POST':
        # selected variable
        selectedvar = request.form.get('selected-var')
        # selected plot type
        selectedplot = request.form.get('plottype')

        # read all files
        df = pd.read_csv(session.get('uploaded_data_file_path'))
        vartype_df = pd.read_csv(session.get('vartype_filepath'))
        # subset dataset
        varclass = vartype_df.loc[vartype_df['column']
                                  == selectedvar, 'class'].values[0]
        vartype = vartype_df.loc[vartype_df['column']
                                 == selectedvar, 'type'].values[0]
        var_df = df[[selectedvar]].copy()
        var_df[selectedvar] = var_df[selectedvar].astype(vartype)
        # remove missing value
        var_df = var_df.dropna(how='any')

        # buffer to store image file
        buf = BytesIO()
        # empty canvas for plotting
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot()

        # numeric variable
        if varclass == 'numeric':
            # density plot
            if selectedplot == 'densityplot':
                dp = ax.hist(var_df, 100, density=True,
                             histtype='stepfilled', facecolor='#5c729f', alpha=0.75)
                var_df[selectedvar].plot.kde(zorder=2, color='#5c729f')
                ax.set_xlim(var_df[selectedvar].min(),
                            var_df[selectedvar].max())
                ax.set_xlabel(selectedvar)
                plt.title('Density Plot')
            # default boxplot
            else:
                bp = ax.boxplot(var_df, vert=False, patch_artist=True)
                bp['boxes'][0].set_facecolor('#5c729f')
                bp['medians'][0].set(color='black')
                ax.set_yticklabels([''])
                ax.set_xlabel(selectedvar)
                plt.title('Box Plot')
        # categorical variable
        else:
            # count of items
            count_df = var_df.value_counts().rename_axis(
                selectedvar).reset_index(name='counts')
            color_values = get_color(count_df.shape[0])
            # pie chart
            if selectedplot == 'piechart':
                wedges, texts = ax.pie(count_df['counts'],
                                       labels=[calculate_percentage(
                                           v, count_df['counts']) for v in count_df['counts']],
                                       labeldistance=1.05,
                                       colors=color_values,
                                       wedgeprops=dict(edgecolor='w', alpha=0.75))
                ax.legend(wedges,
                          labels=count_df[selectedvar],
                          title=selectedvar,
                          loc="upper left",
                          bbox_to_anchor=(1, 1))
                plt.title('Pie Chart')
            # bar plot
            else:
                ax.barh(count_df[selectedvar], count_df['counts'],
                        color=color_values, alpha=0.75)
                ax.set_ylabel(selectedvar)
                plt.title('Bar Plot')

        plt.tight_layout()
        plt.savefig(buf, format='png')
        plot_url = base64.b64encode(buf.getbuffer()).decode("ascii")
        return render_template('univariateplot.html', var=selectedvar, varclass=varclass, selectedplot=selectedplot, plot_url=plot_url)
    # restart session
    else:
        return redirect('/')


# Data Transformation
@ app.route('/datatransform', methods=['GET', 'POST'])
def datatransform():
    # current session
    if request.method == 'POST':
        # read all files
        df = pd.read_csv(session.get('uploaded_data_file_path'))
        vartype_df = pd.read_csv(session.get('vartype_filepath'))
        excludedvar_df = pd.read_csv(session.get('excludedvar_filepath'))
        includedvar = excludedvar_df.loc[~excludedvar_df.exclude, 'column'].tolist(
        )
        # user redirected to datatransform.html
        if request.form.get('redirect') == 'datatransform':
            return render_template('datatransform.html', allformula=generate_formula('both'), variables=includedvar, submit='no')
        # user submitted form to select variable
        elif request.form.get('submit') == 'Submit':
            selectedvar = request.form.get('selected-var')
            vartype = vartype_df.loc[vartype_df['column']
                                     == selectedvar, 'class'].values[0]
            formula = generate_formula(vartype)
            # if not contain NA, remove last two method related to NA
            if not df[selectedvar].isna().any():
                del formula[-2:]
            # user hasnt select method
            if not request.form.get('selected-method'):
                return render_template('datatransform.html', allformula=generate_formula('both'), variables=[selectedvar],
                                       vartype=vartype,
                                       formula=formula,
                                       submit='no')
            else:
                selectedmethod = request.form.get('selected-method')
                return render_template('datatransform.html', allformula=generate_formula('both'), variables=[selectedvar], vartype=vartype,
                                       formula=[selectedmethod],
                                       submit='yes')
    # restart session
    else:
        return redirect('/')


# Transformed Data
@ app.route('/datatransformed', methods=['GET', 'POST'])
def datatransformed():
    # current session
    if request.method == 'POST':
        # read all files
        df = pd.read_csv(session.get('uploaded_data_file_path'))
        vartype_df = pd.read_csv(session.get('vartype_filepath'))
        excludedvar_df = pd.read_csv(session.get('excludedvar_filepath'))
        # get form parameters
        selectedvar = request.form.get('selected-var')
        selectedmethod = request.form.get('selected-method')
        # transform variable
        if (selectedmethod == 'Log-transformation') or (selectedmethod == 'Exponential-transformation') or (selectedmethod == 'Standardization'):
            if selectedmethod == 'Log-transformation':
                # check if data is positive
                if df[selectedvar].min() <= 0:
                    return render_template('datatransformed.html', selectedvar=selectedvar, selectedmethod=selectedmethod,
                                           message="Note: Log-transformation is only allowed for positive numerical data.")
                else:
                    newvar = selectedvar + '_log'
                    df[newvar] = numpy.log(df[selectedvar])
            elif selectedmethod == 'Exponential-transformation':
                # check if data is suitable for exp
                if numpy.exp(df[selectedvar]).mean() == math.inf:
                    return render_template('datatransformed.html', selectedvar=selectedvar, selectedmethod=selectedmethod,
                                           message="Note: Exponential-transformation is not suitable for numerical data with large values.")
                newvar = selectedvar + '_exp'
                df[newvar] = numpy.exp(df[selectedvar])
            else:
                scaler = StandardScaler()
                newvar = selectedvar + '_stdscaler'
                df[newvar] = scaler.fit_transform(df[[selectedvar]].values)
            new_row = {'column': [newvar], 'class': [
                'numeric'], 'type': ['float']}
            write_files(df, vartype_df, excludedvar_df, new_row, newvar)
            # filter missing value
            filtered_df = df[[selectedvar, newvar]].copy()
            filtered_df = filtered_df.dropna(how='any')
            # create density plot
            buf = BytesIO()
            [[ax, bx]] = filtered_df.plot.kde(y=[selectedvar, newvar],
                                              zorder=2, color=['#5c729f', '#9f5c72'],
                                              subplots=True, layout=(1, 2), sharex=False, legend=False, figsize=(6, 4))
            ax.hist(filtered_df[[selectedvar]], 100, density=True,
                    histtype='stepfilled', facecolor='#5c729f', alpha=0.75)
            ax.set_xlim(filtered_df[selectedvar].min(),
                        filtered_df[selectedvar].max())
            ax.set_xlabel(selectedvar)
            bx.hist(filtered_df[[newvar]], 100, density=True,
                    histtype='stepfilled', facecolor='#9f5c72', alpha=0.75)
            bx.set_xlim(filtered_df[newvar].min(), filtered_df[newvar].max())
            bx.set_xlabel(newvar)
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plot_url = base64.b64encode(buf.getbuffer()).decode("ascii")
            return render_template('datatransformed.html', selectedvar=selectedvar, newvar=newvar, selectedmethod=selectedmethod, plot_url=plot_url)
        elif (selectedmethod == 'Min-Max_Scaler'):
            scaler = MinMaxScaler()
            newvar = selectedvar + '_minmaxscaler'
            df[newvar] = scaler.fit_transform(df[[selectedvar]].values)
            new_row = {'column': [newvar], 'class': [
                'numeric'], 'type': ['float']}
            write_files(df, vartype_df, excludedvar_df, new_row, newvar)
            # filter missing value
            filtered_df = df[[selectedvar, newvar]].copy()
            filtered_df = filtered_df.dropna(how='any')
            # create boxplot
            buf = BytesIO()
            fig, ax = plt.subplots(nrows=1, ncols=2)
            bp1 = ax[0].boxplot(filtered_df[[selectedvar]],
                                vert=False, patch_artist=True)
            bp1['boxes'][0].set_facecolor('#5c729f')
            bp1['medians'][0].set(color='black')
            ax[0].set_yticklabels([''])
            ax[0].set_xlabel(selectedvar)
            bp2 = ax[1].boxplot(
                filtered_df[[newvar]], vert=False, patch_artist=True)
            bp2['boxes'][0].set_facecolor('#9f5c72')
            bp2['medians'][0].set(color='black')
            ax[1].set_yticklabels([''])
            ax[1].set_xlabel(newvar)
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plot_url = base64.b64encode(buf.getbuffer()).decode("ascii")
            return render_template('datatransformed.html', selectedvar=selectedvar, newvar=newvar, selectedmethod=selectedmethod, plot_url=plot_url)
        elif selectedmethod == 'Binning':
            min_val = df[selectedvar].min()
            max_val = df[selectedvar].max()
            mid_val = (max_val - min_val)/2
            range_values = (min_val, mid_val, max_val)
            # user hasnt submit the cut points
            if not request.form.get('cut'):
                return render_template('datatransformed.html', selectedvar=selectedvar, selectedmethod=selectedmethod, range=range_values, submit='no')
            else:
                # check the cut points
                cut = request.form.get('cut')
                bins = cut.split(',')
                bins = list(map(int, bins))
                # all values fall within the range
                criteria1 = all([True if (bin >= min_val) and (
                    bin <= max_val) else False for bin in bins])
                # cut points are sorted in ascending order
                criteria2 = sorted(bins) == bins
                if criteria1 and criteria2:
                    # make sure bins cover min and max value
                    if bins[0] > min_val:
                        bins.insert(0, min_val-1)
                    if bins[-1] < max_val:
                        bins.append(max_val)
                    # create new categorical variable
                    newvar = selectedvar + '_bin'
                    df[newvar] = pd.cut(df[selectedvar], bins)
                    new_row = {'column': [newvar], 'class': [
                        'categorical'], 'type': ['str']}
                    write_files(df, vartype_df, excludedvar_df,
                                new_row, newvar)
                    # label and count of each category
                    labels = list(df[newvar].unique())
                    counts = list(df[newvar].value_counts())
                    label_count = sorted(
                        list(zip(labels, counts)), key=lambda pair: pair[0])
                    return render_template('datatransformed.html', selectedvar=selectedvar, newvar=newvar, selectedmethod=selectedmethod, labels=label_count, submit='yes')
                else:
                    message = []
                    if not criteria1:
                        message.append(
                            'All cut points should fall within the range.')
                    if not criteria2:
                        message.append(
                            'All cut points have to be listed in ascending order.')
                    return render_template('datatransformed.html', selectedvar=selectedvar, selectedmethod=selectedmethod, range=range_values, message=message, submit='no')
        elif selectedmethod == 'Renaming_Label':
            original_labels = list(df[selectedvar].unique())
            cleaned_labels = [label.replace(' ', '')
                              for label in original_labels]
            # user hasnt rename labels
            if not request.form.get('rename'):
                return render_template('datatransformed.html', selectedvar=selectedvar, selectedmethod=selectedmethod, labels=cleaned_labels, submit='no')
            else:
                if selectedvar[len(selectedvar)-4:len(selectedvar)] == '_bin':
                    newvar = selectedvar
                else:
                    newvar = selectedvar + '_renamed'
                df[newvar] = df[selectedvar]
                new_labels = []
                # replace label
                for i in range(len(original_labels)):
                    new_label = request.form.get(cleaned_labels[i])
                    # no new label given
                    if new_label == '':
                        new_label = cleaned_labels[i]
                    new_labels.append(new_label.replace(' ', ''))
                    df[newvar] = df[newvar].replace(
                        original_labels[i], new_label)
                all_labels = list(zip(cleaned_labels, new_labels))
                # update data file
                new_row = {'column': [newvar], 'class': [
                    'categorical'], 'type': ['str']}
                write_files(df, vartype_df, excludedvar_df, new_row, newvar)
                return render_template('datatransformed.html', selectedvar=selectedvar, selectedmethod=selectedmethod,
                                       all_labels=all_labels, submit='yes')
        elif selectedmethod == 'Outlier-Anomaly_Treatment':
            min_val = df[selectedvar].min()
            max_val = df[selectedvar].max()
            # user hasnt submit replace values
            if not request.form.get('replace'):
                return render_template('datatransformed.html', selectedvar=selectedvar, selectedmethod=selectedmethod, range=(min_val, max_val), submit='no')
            else:
                selectedvalue = float(request.form.get('replace-value'))
                selectedrange = request.form.get('replace-range')
                newvar = selectedvar + '_outlier'
                df[newvar] = df[selectedvar]
                if selectedrange == 'lower':
                    cond = df[newvar] <= selectedvalue
                elif selectedrange == 'upper':
                    cond = df[newvar] >= selectedvalue
                else:
                    cond = df[newvar] == selectedvalue
                # replace value with NaN
                df.loc[cond, newvar] = math.nan
                new_row = {'column': [newvar], 'class': ['numeric'], 'type': [
                    vartype_df.loc[vartype_df['column'] == selectedvar, 'type'].values[0]]}
                write_files(df, vartype_df, excludedvar_df, new_row, newvar)
                # count NaN
                count_na = df[newvar].isnull().sum()
                # create boxplot
                buf = BytesIO()
                fig, ax = plt.subplots(nrows=1, ncols=2)
                bp1 = ax[0].boxplot(df[[selectedvar]].dropna(how='any'),
                                    vert=False, patch_artist=True)
                bp1['boxes'][0].set_facecolor('#5c729f')
                bp1['medians'][0].set(color='black')
                ax[0].set_yticklabels([''])
                ax[0].set_xlabel(selectedvar)
                bp2 = ax[1].boxplot(
                    df[[newvar]].dropna(how='any'), vert=False, patch_artist=True)
                bp2['boxes'][0].set_facecolor('#9f5c72')
                bp2['medians'][0].set(color='black')
                ax[1].set_yticklabels([''])
                ax[1].set_xlabel(newvar)
                plt.tight_layout()
                plt.savefig(buf, format='png')
                plot_url = base64.b64encode(buf.getbuffer()).decode("ascii")
                return render_template('datatransformed.html', selectedvar=selectedvar, selectedmethod=selectedmethod, count_na=count_na, plot_url=plot_url, submit='yes')
        elif selectedmethod == 'Replace_Missing_Value':
            vartype = vartype_df.loc[vartype_df['column']
                                     == selectedvar, 'class'].values[0]
            # user hasnt submit replace value
            if not request.form.get('replace'):
                if vartype == 'numeric':
                    avg = round(df[selectedvar].mean(), 2)
                    return render_template('datatransformed.html', selectedvar=selectedvar, selectedmethod=selectedmethod, vartype=vartype, submit='no', avg=avg)
                else:
                    mode = df[selectedvar].value_counts().idxmax()
                    return render_template('datatransformed.html', selectedvar=selectedvar, selectedmethod=selectedmethod, vartype=vartype, submit='no', mode=mode)

            else:
                # convert replace value to correct data type
                replacevalue = request.form.get('replace-value')
                if vartype == 'numeric':
                    replacevalue = float(replacevalue)

                else:
                    replacevalue = replacevalue.replace(' ', '')

                # count NaN
                count_na = df[selectedvar].isnull().sum()
                # replace NaN
                newvar = selectedvar + '_nan'
                df[newvar] = df[selectedvar]
                df.loc[df[newvar].isna(), newvar] = replacevalue
                new_row = {'column': [newvar], 'class': [vartype], 'type': [
                    vartype_df.loc[vartype_df['column'] == selectedvar, 'type'].values[0]]}
                write_files(df, vartype_df, excludedvar_df, new_row, newvar)
                return render_template('datatransformed.html', selectedvar=selectedvar, selectedmethod=selectedmethod, count_na=count_na, replacevalue=replacevalue, submit='yes')
        elif selectedmethod == 'Delete_Missing_Value':
            # count NaN
            count_na = df[selectedvar].isnull().sum()
            newsize = df.shape[0] - count_na
            # user hasnt submit form to delete NaN
            if not request.form.get('delete'):
                return render_template('datatransformed.html', selectedvar=selectedvar, selectedmethod=selectedmethod, count_na=count_na, newsize=newsize, submit='no')
            else:
                # drop row with NaN in selected variable
                df = df.dropna(axis=0, how='all', subset=[selectedvar])
                df.to_csv(session.get('uploaded_data_file_path'), index=False)
                return render_template('datatransformed.html', selectedvar=selectedvar, selectedmethod=selectedmethod, count_na=count_na, newsize=df.shape[0], submit='yes')

    # restart session
    else:
        return redirect('/')


# Select variables for multivariate plot
@ app.route('/selectmultivar', methods=['GET', 'POST'])
def selectmultivar():
    # current session
    if request.method == 'POST':
        # reset figsize
        session['fig_sizex'] = 7
        session['fig_sizey'] = 4

        # read files
        df = pd.read_csv(session.get('uploaded_data_file_path'))
        vartype_df = pd.read_csv(session.get('vartype_filepath'))
        excludedvar_df = pd.read_csv(session.get('excludedvar_filepath'))
        includedvar = excludedvar_df.loc[~excludedvar_df['exclude'], 'column'].tolist(
        )
        # xvar not chosen
        if not request.form.get('selectedx'):
            return render_template('selectmultivar.html', allplots=generate_plottype('all'), xvar=includedvar, submit='no')
        # xvar chosen
        else:
            xvar = request.form.get('xvar')
            includedvar = [var for var in includedvar if var != xvar]
            # yvar not chosen
            if not request.form.get('selectedy'):
                return render_template('selectmultivar.html', allplots=generate_plottype('all'), xvar=[xvar], yvar=includedvar, submit='no')
            # xvar and yvar chosen
            else:
                yvar = request.form.get('yvar')
                xtype = vartype_df.loc[vartype_df['column']
                                       == xvar, 'class'].values[0]
                ytype = vartype_df.loc[vartype_df['column']
                                       == yvar, 'class'].values[0]
                # zvar not chosen
                if not request.form.get('selectedz'):
                    if (xtype == 'numeric') and (ytype == 'numeric'):
                        includedvar = [
                            var for var in includedvar if var != yvar]
                        includedvar = [
                            var for var in includedvar if var in vartype_df.loc[vartype_df['class'] == 'categorical', 'column'].to_list()]
                        includedvar = [
                            i for i in includedvar if df[i].nunique() < 5]
                        includedvar.append('No-grouping')
                    # x and y not both categorical
                    elif not (xtype == 'categorical' and ytype == 'categorical'):
                        includedvar = [
                            var for var in includedvar if var != yvar]
                        includedvar = [
                            i for i in includedvar if df[i].nunique() < 5]
                        includedvar.append('No-grouping')
                        # unique items in group no greater than 5, to avoid messy chart
                    else:
                        includedvar = ['noz']
                    return render_template('selectmultivar.html', allplots=generate_plottype('all'), xvar=[xvar], yvar=[yvar], zvar=includedvar, submit='no')
                # xvar, yvar and zvar chosen
                else:
                    zvar = request.form.get('zvar')
                    # plot not chosen
                    if not request.form.get('selectedplot'):
                        plots = generate_plottype(xtype+'-'+ytype)
                        plots = [plot[0] for plot in plots]
                        # exclude count-plot if yvar has more than 2 levels
                        if (xtype == 'categorical') and (ytype == 'categorical'):
                            if (df[yvar].nunique() > 2) and (df[xvar].nunique() > 2):
                                plots.remove('count-plot')
                        if (xtype == 'numeric') and (ytype == 'categorical') and (zvar != 'No-grouping'):
                            plots.remove('density-plot')
                        # no available plot
                        if len(plots) == 0:
                            plots = ['No-plot']
                        return render_template('selectmultivar.html', allplots=generate_plottype('all'), xvar=[xvar], yvar=[yvar], zvar=[zvar], plots=plots, submit='no')
                    # all variables and plot chosen
                    else:
                        plot = request.form.get('plot')
                        return render_template('selectmultivar.html', xvar=xvar, yvar=yvar, zvar=zvar, plot=plot, submit='yes')
    # restart session
    else:
        return redirect('/')


# Correlation plot
@ app.route('/heatmap', methods=['GET', 'POST'])
def heatmap():
    # current session
    if request.method == 'POST':
        # file path
        file_path = session.get('uploaded_data_file_path')
        # read data file
        df = pd.read_csv(file_path)
        # read variable type file
        vartype = pd.read_csv(session.get('vartype_filepath'))
        numericvar = vartype.loc[vartype['class']
                                 == 'numeric', 'column'].values.tolist()
        # read excluded variable file
        excludedvar_df = pd.read_csv(session.get('excludedvar_filepath'))
        excludedvar = excludedvar_df.loc[excludedvar_df.exclude,
                                         'column'].values.tolist()

        selectedvar = [var for var in numericvar if var not in excludedvar]

        # no numerical variables
        if len(selectedvar) == 0:
            return render_template('heatmap.html', nvar=0)
        else:
            new_df = df[selectedvar].copy()
            # update variable type
            for col in selectedvar:
                if vartype.loc[vartype['column'] == col, 'type'].values[0] == 'int':
                    new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                    # create a boolean mask for not NA values
                    mask = new_df[col].notna()
                    # convert not NA values to integer
                    new_df.loc[mask, col] = new_df.loc[mask,
                                                       col].astype(numpy.int64)
                else:
                    new_df[col] = new_df[col].astype(numpy.float64)

            # buffer to store image file
            buf = BytesIO()
            # empty canvas for plotting
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
            # heatmap
            sns.heatmap(new_df.corr(), annot=True, linewidths=.5, cmap="Blues")
            plt.title('Heatmap of correlations between numerical data')
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plot_url = base64.b64encode(buf.getbuffer()).decode("ascii")
            return render_template('heatmap.html', nvar=len(selectedvar), plot_url=plot_url)
    else:
        return redirect('/')


# Multivariate plot
@ app.route('/multivariateplot', methods=['GET', 'POST'])
def multivariateplot():
    # current session
    if request.method == 'POST':
        # read all files
        df = pd.read_csv(session.get('uploaded_data_file_path'))
        vartype_df = pd.read_csv(session.get('vartype_filepath'))

        # get form parameters
        xvar = request.form.get('xvar')
        xtype = vartype_df.loc[vartype_df['column'] == xvar, 'class'].values[0]
        yvar = request.form.get('yvar')
        ytype = vartype_df.loc[vartype_df['column'] == yvar, 'class'].values[0]
        zvar = request.form.get('zvar')
        plot = request.form.get('plot')

        # subset data set
        if (zvar != 'noz') and (zvar != 'No-grouping'):
            df = df[[xvar, yvar, zvar]].copy()
        else:
            df = df[[xvar, yvar]]
        # convert variable type
        for col in df.columns.tolist():
            vartype = vartype_df.loc[vartype_df['column']
                                     == col, 'type'].values[0]
            df[col] = df[col].astype(vartype)

        # buffer to store image file
        buf = BytesIO()

        # adjust figure size in x,y dimension
        if request.form.get('changexy'):
            sizex = session.get('fig_sizex')
            sizey = session.get('fig_sizey')
            if request.form.get('changexy') == "Expand X axis":
                session['fig_sizex'] = sizex + 1
            else:
                session['fig_sizey'] = sizey + 1

        if request.form.get('resetsize'):
            session['fig_sizex'] = 7
            session['fig_sizey'] = 4

        if request.form.get('switch'):
            xvar, yvar = yvar, xvar
            xtype, ytype = ytype, xtype
            # session['fig_sizex'], session['fig_sizey'] = session['fig_sizey'], session['fig_sizex']

        # empty canvas for plotting
        fig = plt.figure(figsize=(session['fig_sizex'], session['fig_sizey']))
        ax = fig.add_subplot()

        # plot
        if plot == 'box-plot':
            if zvar != 'noz' and zvar != 'No-grouping':
                sns.boxplot(data=df, x=xvar, y=yvar,
                            hue=zvar, palette='rainbow')
            else:
                sns.boxplot(data=df, x=yvar, y=xvar, palette='rainbow')

        elif plot == 'bar-plot_sum':
            if zvar != 'noz' and zvar != 'No-grouping':
                if xtype == 'numeric':
                    aggregated = df.groupby([yvar, zvar])[
                        xvar].sum().reset_index()
                    agg_sort = aggregated.sort_values(xvar, ascending=False)
                else:
                    aggregated = df.groupby([xvar, zvar])[
                        yvar].sum().reset_index()
                    agg_sort = aggregated.sort_values(yvar, ascending=False)

                if request.form.get('sort'):
                    sns.barplot(data=agg_sort, x=xvar, y=yvar, hue=zvar,
                                errwidth=0, palette='rainbow')
                else:
                    sns.barplot(data=aggregated, x=xvar, y=yvar, hue=zvar,
                                errwidth=0, palette='rainbow')

            else:
                if xtype == 'numeric':
                    aggregated = df.groupby(yvar)[xvar].sum().reset_index()
                    agg_sort = aggregated.sort_values(xvar, ascending=False)
                else:
                    aggregated = df.groupby(xvar)[yvar].sum().reset_index()
                    agg_sort = aggregated.sort_values(yvar, ascending=False)

                if request.form.get('sort'):
                    sns.barplot(data=agg_sort, x=xvar, y=yvar,
                                errwidth=0, palette='rainbow',)
                else:
                    sns.barplot(data=aggregated, x=xvar, y=yvar,
                                errwidth=0, palette='rainbow',)

        elif plot == 'bar-plot_average':
            if zvar != 'noz' and zvar != 'No-grouping':
                if xtype == 'numeric':
                    aggregated = df.groupby([yvar, zvar])[
                        xvar].mean().reset_index()
                    agg_sort = aggregated.sort_values(xvar, ascending=False)
                else:
                    aggregated = df.groupby([xvar, zvar])[
                        yvar].mean().reset_index()
                    agg_sort = aggregated.sort_values(yvar, ascending=False)

                if request.form.get('sort'):
                    sns.barplot(data=agg_sort, x=xvar, y=yvar, hue=zvar,
                                errwidth=0, palette='rainbow')
                else:
                    sns.barplot(data=aggregated, x=xvar, y=yvar, hue=zvar,
                                errwidth=0, palette='rainbow')

            else:
                if xtype == 'numeric':
                    aggregated = df.groupby(yvar)[xvar].mean().reset_index()
                    agg_sort = aggregated.sort_values(xvar, ascending=False)
                else:
                    aggregated = df.groupby(xvar)[yvar].mean().reset_index()
                    agg_sort = aggregated.sort_values(yvar, ascending=False)

                if request.form.get('sort'):
                    sns.barplot(data=agg_sort, x=xvar, y=yvar,
                                errwidth=0, palette='rainbow',)
                else:
                    sns.barplot(data=aggregated, x=xvar, y=yvar,
                                errwidth=0, palette='rainbow',)

        elif plot == 'density-plot':
            if xtype == 'numeric':
                sns.kdeplot(data=df, x=xvar, hue=yvar, fill=True,
                            common_norm=False, palette='rainbow', alpha=.5, linewidth=1)
            else:
                sns.kdeplot(data=df, x=yvar, hue=xvar, fill=True,
                            common_norm=False, palette='rainbow', alpha=.5, linewidth=1)

        elif plot == 'count-plot':
            if request.form.get('sort'):
                # if df[xvar].nunique() == 2:
                #     sns.countplot(
                #         data=df, y=yvar, hue=xvar, palette='rainbow', order=df[yvar].value_counts().index)
                # else:
                sns.countplot(
                    data=df, y=xvar, hue=yvar, palette='rainbow', order=df[xvar].value_counts().index)

            else:
                # if df[xvar].nunique() == 2:
                #     sns.countplot(data=df, y=yvar, hue=xvar, palette='rainbow')
                # else:
                sns.countplot(data=df, y=xvar, hue=yvar, palette='rainbow')

        elif plot == 'scatter-plot':
            if (zvar == 'noz') or (zvar == 'No-grouping'):
                sns.scatterplot(data=df, x=xvar, y=yvar, palette='rainbow')
            else:
                sns.scatterplot(data=df, x=xvar, y=yvar,
                                hue=zvar, style=zvar, palette='rainbow')
        elif plot == 'line-plot':
            if (zvar == 'noz') or (zvar == 'No-grouping'):
                sns.lineplot(data=df, x=xvar, y=yvar, palette='rainbow')
            else:
                sns.lineplot(data=df, x=xvar, y=yvar,
                             hue=zvar, palette='rainbow')

        plt.tight_layout()
        plt.savefig(buf, format='png')
        plot_url = base64.b64encode(buf.getbuffer()).decode("ascii")

        variables = ", ".join(df.columns.tolist())
        return render_template('multivariateplot.html', xvar=xvar, yvar=yvar, zvar=zvar, plot=plot, variables=variables, plot_url=plot_url)
    # restart session
    else:
        return redirect('/')
