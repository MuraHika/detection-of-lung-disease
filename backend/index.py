from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageDraw
from predict import main

app = Flask(__name__)

app.config.from_object(__name__)

CORS(app, resources={r"/*":{'origins':"*"}})

@app.route('/', methods=['POST'])
def get_image():
    file = request.files['image']
    print(file)
    path = image_processing(file)
    results = ''
    for desiase in ['covid', 'pneumonia', 'tuberculosis']:
        diagnosis = main(path, desiase)
        results += diagnosis + ', '
    return results

def image_processing(image):
    img = Image.open(image)
    bw = img.convert('L')
    path = 'x_ray.png'
    bw.save(path)
    return path

if __name__ == '__main__':
    app.run(debug=True)