# -----------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# -----------------------------------------------------------------------------------------


from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        return render_template("index.html", shape=df.shape, tables=[df.head().to_html()], titles=[''])
    return render_template("index.html")
