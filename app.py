# -----------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# -----------------------------------------------------------------------------------------

# Required libraries
from flask import Flask, render_template, request, session
import pandas as pd
import os
from werkzeug.utils import secure_filename


# Flask configuration
upload_folder = os.path.join('static', 'uploadfile')
filelist = [file for file in os.listdir(upload_folder)]
for file in filelist:
    os.remove(os.path.join(upload_folder, file))
allowed_extension = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder
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
            df = pd.read_csv(session.get('uploaded_data_file_path'))
            if df.shape[0] >= 2 and df.shape[1] >= 2:
                return render_template('index.html', success="yes")
            else:
                os.remove(session.get('uploaded_data_file_path'))
                session['uploaded_data_file_path'] = ""
                return render_template('index.html', success="no")


# Select variable
@app.route('/selectvariable', methods=['GET', 'POST'])
def selectVariable():
    # file path
    file_path = session.get('uploaded_data_file_path', None)
    # read csv file
    df = pd.read_csv(file_path)

    if request.method == 'GET':
        return render_template('selectvariable.html', shape=df.shape, columns=df.columns, error=False)

    else:
        x = request.form.get('xvar')
        x_measure = request.form.get('xmeasure')
        y = request.form.get('yvar')
        # check if x and y are different
        if x == y:
            return render_template('selectvariable.html', shape=df.shape, columns=df.columns, error=True)
        else:
            return render_template('selectedvariable.html', x=x, x_measure=x_measure, y=y)


@app.route('/selectedvariable', methods=['GET', 'POST'])
def selectedVariable():
    return render_template('selectedvariable.html')
