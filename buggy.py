from flask import Flask, render_template, jsonify, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from loadH5 import runPrediction
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

@app.route('/', methods=["GET", "POST"])
def upload_file():

    if request.method == 'POST':
    
        if request.files:
            image = request.files['inputFile']
            filename = secure_filename(image.filename)

            image.save(os.path.join(os.getcwd(), image.filename))

            runPrediction(image.filename)
            return redirect(request.url)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)