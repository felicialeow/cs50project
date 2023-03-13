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
                    'method',
                    'variable']
    for info in list_of_info:
        session[info] = ""


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


# Helper function 6: generate list of formula for feature engineering
def generate_formula(vartype):
    if vartype == 'numeric':
        formula = ['Take_log',
                   'Take_exponential',
                   'Min-Max_Scaler',
                   'Standardization',
                   'Binning']
    else:
        formula = ['Rename_label']
    return formula


# Helper function 7: write files
def write_files(df, vartype_df, excludedvar_df, new_row, newvar):
    vartype_df = vartype_df.append(new_row, ignore_index=True)
    excludedvar_df = excludedvar_df.append(
        {'column': newvar, 'exclude': False}, ignore_index=True)
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
        html_table = df.head().to_html(index=False, header=True, classes='table-style')
        colnames = df.columns.tolist()
        # read variable type file
        if request.form.get('reset') == 'datatype':
            vartype = pd.read_csv(session.get('vartype_FIXED_filepath'))
        else:
            vartype = pd.read_csv(session.get('vartype_filepath'))
        vartype_list = list(zip(vartype['column'], vartype['class']))

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
                    new_df[col] = new_df[col].astype(numpy.int64)
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
                [df_desc, new_df.isnull().sum(axis=0).to_frame('NA').transpose()])
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
@app.route('/selectvariable', methods=['GET', 'POST'])
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
@app.route('/univariateplot', methods=['GET', 'POST'])
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
@app.route('/datatransform', methods=['GET', 'POST'])
def datatransform():
    # read all files
    df = pd.read_csv(session.get('uploaded_data_file_path'))
    vartype_df = pd.read_csv(session.get('vartype_filepath'))
    excludedvar_df = pd.read_csv(session.get('excludedvar_filepath'))

    includedvar = excludedvar_df.loc[~excludedvar_df.exclude, 'column'].tolist(
    )

    if request.method == 'GET':
        if request.args.get('selected-var1'):
            selectedvar = request.args.get('selected-var1')
            vartype = vartype_df.loc[vartype_df['column']
                                     == selectedvar, 'class'].values[0]
            if request.args.get('selected-method'):
                selectedmethod = request.args.get('selected-method')

                if selectedmethod == "Take_log":
                    newvar = selectedvar + '_log'
                    df[newvar] = numpy.log(df[selectedvar])
                    new_row = {'column': newvar,
                               'class': 'numeric', 'type': 'float'}
                    write_files(df, vartype_df, excludedvar_df,
                                new_row, newvar)

                if selectedmethod == "Take_exponential":
                    newvar = selectedvar + '_exp'
                    df[newvar] = numpy.exp(df[selectedvar])
                    new_row = {'column': newvar,
                               'class': 'numeric', 'type': 'float'}
                    write_files(df, vartype_df, excludedvar_df,
                                new_row, newvar)

                if selectedmethod == "Min-Max_Scaler":
                    scaler = MinMaxScaler()
                    newvar = selectedvar + '_minmaxscaler'
                    df[newvar] = scaler.fit_transform(
                        df[selectedvar].values.reshape(-1, 1))
                    new_row = {'column': newvar,
                               'class': 'numeric', 'type': 'float'}
                    write_files(df, vartype_df, excludedvar_df,
                                new_row, newvar)

                if selectedmethod == "Standardization":
                    scaler = StandardScaler()
                    newvar = selectedvar + '_stdscaler'
                    df[newvar] = scaler.fit_transform(
                        df[selectedvar].values.reshape(-1, 1))
                    new_row = {'column': newvar,
                               'class': 'numeric', 'type': 'float'}

                    write_files(df, vartype_df, excludedvar_df,
                                new_row, newvar)

                if selectedmethod == "Binning":
                    session['method'] = selectedmethod
                    session['variable'] = selectedvar
                    return render_template("binning.html", method=selectedmethod, selectedvar=selectedvar, done='No')

                if selectedmethod == "Rename_label":
                    session['method'] = selectedmethod
                    session['variable'] = selectedvar
                    labels = df[selectedvar].unique()
                    return render_template("rename_label.html", method=selectedmethod, selectedvar=selectedvar, done='No', labels=labels)

                return render_template('newfeature.html', vars=None, method=selectedmethod, selectedvar=selectedvar, newcreated=newvar)
            else:
                return render_template('newfeature.html', vars=None, method=None, selectedvar=selectedvar, vartype=vartype, options=generate_formula(vartype))
        else:
            return render_template('newfeature.html', vars=includedvar)


# Select multivariable
@app.route('/selectmultivar', methods=['GET', 'POST'])
def selectmultivar():

    # read files
    vartype_df = pd.read_csv(session.get('vartype_filepath'))
    excludedvar_df = pd.read_csv(session.get('excludedvar_filepath'))
    includedvar = excludedvar_df.loc[~excludedvar_df['exclude'], 'column'].tolist(
    )
    # categorical variables for x and group
    categoricalvar = [
        col for col in includedvar if col in vartype_df.loc[vartype_df['class'] == 'categorical', 'column'].values]
    # numeric variables for y
    numericvar = [
        col for col in includedvar if col in vartype_df.loc[vartype_df['class'] == 'numeric', 'column'].values]

    return render_template('selectmultivar.html', xs=includedvar, ys=numericvar, groups=categoricalvar)


# Multivariate plot
# https://www.kaggle.com/code/alokevil/simple-eda-for-beginners


@app.route('/multivariateplot', methods=['GET', 'POST'])
def multivariateplot():
    x = request.form.get('selected_x')
    y = request.form.get('selected_y')
    z = request.form.get('selected_group')

    if x is not None:
        session['x'] = x
    else:
        x = session.get('x')

    if y is not None:
        session['y'] = y
    else:
        y = session.get('y')

    if z is not None:
        session['z'] = z
    else:
        z = session.get('z')

    # read files
    vartype_df = pd.read_csv(session.get('vartype_filepath'))
    excludedvar_df = pd.read_csv(session.get('excludedvar_filepath'))
    includedvar = excludedvar_df.loc[~excludedvar_df['exclude'], 'column'].tolist(
    )
    categoricalvar = [
        col for col in includedvar if col in vartype_df.loc[vartype_df['class'] == 'categorical', 'column'].values]

    numericvar = [
        col for col in includedvar if col in vartype_df.loc[vartype_df['class'] == 'numeric', 'column'].values]

    # some inputs are same
    if x == y or x == z:
        return render_template('error.html', msg="Error:Please make sure inputs are different")
    if x in numericvar and y == "_count_":
        return render_template('error.html', msg="Error:Cannot plot count when x is numeric")

    # if z == "none" and y == "_count_":
    #     return render_template('error.html', msg="Error:Cannot plot count when group is none or x and group are both categorical")

    file_path = session.get('uploaded_data_file_path')
    # read all files
    df = pd.read_csv(file_path)

    # buffer to store image file
    buf = BytesIO()
    # # empty canvas for plotting
    if request.form.get('expandx'):
        xaxis = session.get('xaxis')
        xaxis = xaxis + 1
        session['xaxis'] = xaxis
    else:
        session['xaxis'] = 8
        xaxis = session.get('xaxis')
    fig = plt.figure(figsize=(xaxis, 6))

    # ax = fig.add_subplot()
    if y == '_count_' and z != 'none':
        sns.countplot(x=x, hue=z, data=df, order=df[x].value_counts().index)

    elif z == 'none':
        if y == '_count_':
            sns.countplot(x=x, data=df, order=df[x].value_counts().index)
        elif x in numericvar and y in numericvar:
            sns.relplot(x=x, y=y, data=df)
        else:
            sns.boxplot(x=x, y=y, data=df)

    else:
        sns.boxplot(x=x, y=y, hue=z, data=df)

    plt.tight_layout()
    plt.savefig(buf, format='png')
    plot_url = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render_template('multivariateplot.html', plot_url=plot_url)


@app.route('/binning', methods=['GET', 'POST'])
def bin():
    # read all files
    df = pd.read_csv(session.get('uploaded_data_file_path'))
    vartype_df = pd.read_csv(session.get('vartype_filepath'))
    excludedvar_df = pd.read_csv(session.get('excludedvar_filepath'))

    selectedmethod = session.get('method')
    selectedvar = session.get('variable')
    cut = request.form.get("cut")

    bins = cut.split(',')
    bins = list(map(int, bins))

    # make sure bins cover min and max value
    if bins[0] > df[selectedvar].min():
        bins.insert(0, int(df[selectedvar].min())-1)
    if bins[-1] < df[selectedvar].max():
        bins.append(int(df[selectedvar].max()))

    newvar = session.get('variable') + '_bin'
    df[newvar] = pd.cut(df[selectedvar], bins)
    new_row = {'column': newvar,
               'class': 'categorical', 'type': 'str'}
    write_files(df, vartype_df, excludedvar_df,
                new_row, newvar)
    return render_template('binning.html', method=selectedmethod, selectedvar=selectedvar, done='Yes', newvar=newvar)


@app.route('/rename_label', methods=['GET', 'POST'])
def relabel():
    # read all files
    # changed_labellist = session.get('temp_list')

    df = pd.read_csv(session.get('uploaded_data_file_path'))

    selectedmethod = session.get('method')
    selectedvar = session.get('variable')

    newlabel = request.form.getlist("newlabel")
    orilabel = request.form.getlist("orilabel")
    # print(newlabel)
    for i, j in zip(newlabel, orilabel):
        if i != '':
            df[selectedvar] = df[selectedvar].replace(j, i)

    df.to_csv(session.get(
        'uploaded_data_file_path'), index=False)

    return render_template('rename_label.html', method=selectedmethod, selectedvar=selectedvar, done='Yes')
