# -----------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# -----------------------------------------------------------------------------------------

# Required libraries
from flask import Flask, render_template, request, session, redirect, render_template_string
import pandas as pd
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy
from scipy.stats import gaussian_kde


# Helper function
# calculate percentage in pie chart
def calculate_percentage(count, allvals):
    percent = numpy.round(count/numpy.sum(allvals)*100)
    return f"{percent:.1f}%"


# get color value for discrete groups
def get_color(n):
    cmap = plt.get_cmap('gist_rainbow', 100)
    color_values = cmap([val*int(100/(n-1)) for val in range(n)])
    return color_values


# Set parameters of all text in figure
smallfontsize = 6
normalfontsize = 8
largefontsize = 10
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


# Clear directory
upload_folder = os.path.join('static', 'uploadfile')
filelist = [file for file in os.listdir(upload_folder)]
for file in filelist:
    if (file == 'sample.csv'):
        continue
    else:
        os.remove(os.path.join(upload_folder, file))
allowed_extension = {'csv'}


# Flask configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder
# app.config['FLAG_CHANGE_DATA'] = 0
app.secret_key = "secret"


# Upload file
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', success="initial")
    else:
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
            # check data structure
            # at least 2 rows and 2 columns
            df = pd.read_csv(session.get('uploaded_data_file_path'))
            if (df.shape[0] >= 2 and df.shape[1] >= 2):
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
                return render_template('index.html', success="yes")
            # remove invalid data
            else:
                os.remove(session.get('uploaded_data_file_path'))
                session['uploaded_data_file_path'] = ""
                return render_template('index.html', success="no")


# Data structure
@app.route('/datatype', methods=['GET', 'POST'])
def datatype():
    # file path
    file_path = session.get('uploaded_data_file_path')
    # read csv file
    df = pd.read_csv(file_path)
    colnames = df.columns.tolist()
    # read variable type file
    vartype = pd.read_csv(session.get('vartype_filepath'))
    vartype_list = list(zip(vartype['column'], vartype['class']))

    if request.method == 'GET':
        return render_template('datatype.html', df=[df.head().to_html()], shape=df.shape, vartype=vartype_list)
    else:
        # update any change in variable type
        for col in colnames:
            newcolclass = request.form.get(col)
            if newcolclass != vartype.loc[vartype['column'] == col, 'class'].values[0]:
                vartype.loc[vartype['column'] == col, 'class'] = newcolclass
                if newcolclass == 'numeric':
                    vartype.loc[vartype['column'] == col, 'type'] = 'int'
                else:
                    vartype.loc[vartype['column'] == col, 'type'] = 'str'
        vartype.to_csv(session.get('vartype_filepath'), index=False)
        # create empty dataframe to store excluded variables
        excludedvar = pd.DataFrame({'column': colnames, 'exclude': False})
        excludedvar.to_csv(os.path.join(
            app.config['UPLOAD_FOLDER'], 'excludedvar.csv'), index=False)
        # store file path of excluded variables in session
        session['excludedvar_filepath'] = os.path.join(
            app.config['UPLOAD_FOLDER'], 'excludedvar.csv')
        # redirect to numerical descriptions
        if any([True if t == 'numeric' else False for t in vartype['class']]):
            return redirect('/numeric')
        else:
            return redirect('/categorical')


# Description of numeric variables
@app.route('/numeric', methods=['GET', 'POST'])
def numeric():
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
    # subset dataset
    selectedvar = [var for var in numericvar if var not in excludedvar]
    new_df = df[selectedvar].copy()
    # update variable type
    for col in selectedvar:
        if vartype.loc[vartype['column'] == col, 'type'].values[0] == 'int':
            new_df[col] = new_df[col].astype(numpy.int64)
        else:
            new_df[col] = new_df[col].astype(numpy.float64)
    # summary statistics of variables
    df_desc = new_df.describe().round(2)

    if request.method == 'GET':
        return render_template('numeric.html', df=[df_desc.to_html(header="true", classes='mystyle')], columns=selectedvar, nvar=df_desc.shape[1])
    else:
        if request.form.get('remove-numericvar'):
            col = request.form.get('remove-numericvar')
            excludedvar_df.loc[excludedvar_df.column == col, 'exclude'] = True
        elif request.form.get('reset-numericvar'):
            excludedvar_df.loc[excludedvar_df.column.isin(numericvar),
                               'exclude'] = False
        excludedvar_df.to_csv(session.get('excludedvar_filepath'), index=False)
        return redirect('/numeric')


# Description of categorical variables
@app.route('/categorical', methods=['GET', 'POST'])
def categorical():
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
    excludedvar = excludedvar_df.loc[excludedvar_df.exclude,
                                     'column'].values.tolist()
    # subset dataset
    selectedvar = [var for var in categoricalvar if var not in excludedvar]
    new_df = df[selectedvar].copy()
    # update variable type
    for col in selectedvar:
        new_df[col] = new_df[col].astype(str)
    # summary statistics of variables
    df_desc = new_df.describe()

    if request.method == 'GET':
        return render_template('categorical.html', df=[df_desc.to_html(header="true", classes='mystyle')], columns=selectedvar, nvar=df_desc.shape[1])
    else:
        if request.form.get('remove-categoricalvar'):
            col = request.form.get('remove-categoricalvar')
            excludedvar_df.loc[excludedvar_df.column == col, 'exclude'] = True
        elif request.form.get('reset-categoricalvar'):
            excludedvar_df.loc[excludedvar_df.column.isin(categoricalvar),
                               'exclude'] = False
        excludedvar_df.to_csv(session.get('excludedvar_filepath'), index=False)
        return redirect('/categorical')


# Select variable
@app.route('/selectvariable', methods=['GET', 'POST'])
def selectvariable():
    # file path
    file_path = session.get('uploaded_data_file_path')
    # read all files
    df = pd.read_csv(file_path)
    vartype_df = pd.read_csv(session.get('vartype_filepath'))
    excludedvar_df = pd.read_csv(session.get('excludedvar_filepath'))
    includedvar = excludedvar_df.loc[~excludedvar_df['exclude'], 'column'].tolist(
    )
    # categorical variables
    categoricalvar = [
        col for col in includedvar if col in vartype_df.loc[vartype_df['class'] == 'categorical', 'column'].values]
    # numeric variables
    numericvar = [
        col for col in includedvar if col in vartype_df.loc[vartype_df['class'] == 'numeric', 'column'].values]

    if request.method == 'GET':
        return render_template('selectvariable.html', selectedtype=None, varoptions=None)
    else:
        selectedtype = request.form.get('selected-datatype')
        if selectedtype == 'numeric':
            return render_template('selectvariable.html', selectedtype=selectedtype, varoptions=numericvar)
        else:
            return render_template('selectvariable.html', selectedtype=selectedtype, varoptions=categoricalvar)


# Univariate plot
@app.route('/univariateplot', methods=['GET', 'POST'])
def univariateplot():
    # file path
    file_path = session.get('uploaded_data_file_path')
    # read all files
    df = pd.read_csv(file_path)
    vartype_df = pd.read_csv(session.get('vartype_filepath'))

    if request.method == 'GET':
        if request.args.get('selected-var'):
            selectedvar = request.args.get('selected-var')
            varclass = vartype_df.loc[vartype_df['column']
                                      == selectedvar, 'class'].values[0]
            vartype = vartype_df.loc[vartype_df['column']
                                     == selectedvar, 'type'].values[0]
            var_df = df[[selectedvar]].copy()
            var_df[selectedvar] = var_df[selectedvar].astype(vartype)
            buf = BytesIO()
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot()

            if varclass == 'numeric':
                if request.args.get('plottype') == 'densityplot':
                    dp = ax.hist(
                        var_df, 100, density=True, histtype='stepfilled', facecolor='#5c729f', alpha=0.75)
                    var_df[selectedvar].plot.kde(zorder=2, color='#5c729f')
                    ax.set_xlim(var_df[selectedvar].min(),
                                var_df[selectedvar].max())
                    ax.set_xlabel(selectedvar)
                    plt.title('Density Plot')
                else:
                    bp = ax.boxplot(var_df, vert=False,
                                    patch_artist=True)
                    bp['boxes'][0].set_facecolor('#5c729f')
                    bp['medians'][0].set(color='black')
                    ax.set_yticklabels([''])
                    ax.set_xlabel(selectedvar)
                    plt.title('Box Plot')
            else:
                count_df = var_df.value_counts().rename_axis(
                    selectedvar).reset_index(name='counts')
                color_values = get_color(count_df.shape[0])
                if request.args.get('plottype') == 'piechart':
                    wedges, texts = ax.pie(count_df['counts'],
                                           labels=[calculate_percentage(
                                               v, count_df['counts']) for v in count_df['counts']],
                                           labeldistance=1.05,
                                           colors=color_values,
                                           wedgeprops=dict(edgecolor='w', alpha=0.75))
                    # plt.setp(texts, fontsize=8)
                    ax.legend(wedges,
                              labels=count_df[selectedvar],
                              title=selectedvar,
                              loc="upper left",
                              bbox_to_anchor=(1, 1))
                    plt.title('Pie Chart')
                else:
                    ax.barh(count_df[selectedvar], count_df['counts'],
                            color=color_values, alpha=0.75)
                    ax.set_ylabel(selectedvar)
                    plt.title('Bar Plot')

            plt.tight_layout()
            plt.savefig(buf, format='png')
            plot_url = base64.b64encode(buf.getbuffer()).decode("ascii")
            return render_template('univariateplot.html', var=selectedvar, varclass=varclass, plot_url=plot_url)


@app.route('/basics', methods=['GET', 'POST'])
def basics():
    # file path
    file_path = session.get('uploaded_data_file_path')
    # temp file [to be removed after web is built]
    temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.csv')

    # if no variable to delete
    if app.config['FLAG_CHANGE_DATA'] == 0:
        df = pd.read_csv(file_path)  # read original file
        df_desc = df.describe().round(2)
    # if need to delete
    else:
        df = pd.read_csv(file_path)  # read original file
        # read temp description file
        df_desc = pd.read_csv(temp_filepath, index_col=0)

    if request.method == 'GET':
        return render_template('basics.html', shape=df.shape, tables=[df_desc.to_html(header="true", classes='mystyle')], cols=df_desc.columns)
    else:
        rvar = request.form.get('rvar')
        # if need to remove variable
        if rvar is not None:
            # remove variable
            df_desc.drop(rvar, axis=1, inplace=True)
            # store in temp file
            df_desc.to_csv(tempfile)
            app.config['FLAG_CHANGE_DATA'] = 1
            return render_template('basics.html', shape=df.shape, tables=[df_desc.to_html(header="true", classes='mystyle')], cols=df_desc.columns)
        # reset
        else:
            app.config['FLAG_CHANGE_DATA'] = 0
            df = pd.read_csv(file_path)
            df_desc = df.describe().round(2)
            return render_template('basics.html', shape=df.shape, tables=[df_desc.to_html(header="true", classes='mystyle')], cols=df_desc.columns)
