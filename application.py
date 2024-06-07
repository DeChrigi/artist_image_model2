from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Überprüft, ob die Datei eine erlaubte Erweiterung hat
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Konvertiert das Bild in ein Format, das das Modell verarbeiten kann
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((150, 150))  # Ändere die Größe entsprechend den Anforderungen deines Modells
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def getArtistCategories():
    dfList = []
    countIndex = 0
    for ordner_name in os.listdir("./dataset/train_images"):
        dfList.append({"id": countIndex, "artist_name": str.replace(ordner_name, '_', ' ')})
        countIndex += 1

    df = pd.DataFrame(dfList, columns=["id", "artist_name"])

    return df

# Definiere die Routen
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Wähle das Modell basierend auf der Benutzerauswahl
        selected_model = request.form['model']
        model_path = f'./trained_models/{selected_model}.h5'
        model = load_model(model_path)
        
        # Preprocess and predict
        image = preprocess_image(filepath)
        prediction = model.predict(image)[0]  # Nimm die erste Vorhersage aus der Batch

        # Künstlers Namen und Wahrscheinlichkeiten basierend auf der vorhergesagten Klasse bestimmen
        artists = getArtistCategories()
        artists['probability'] = prediction

        # Sortiere die Künstler nach Wahrscheinlichkeit und begrenze auf die Top 5
        top_artists = artists.sort_values(by='probability', ascending=False).head(5)

        return render_template('result.html', artists=top_artists, filename=filename)
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
