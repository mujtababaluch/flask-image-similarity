from flask import Flask, request, jsonify, render_template
import os
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

# define your image similarity model here



# define the route for your web interface
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare_images', methods=['POST'])
def compare_images():
    # get the uploaded images
    image1 = request.files.get('image1')
    image2 = request.files.get('image2')

    # process the images
    image1 = Image.open(image1)
    image2 = Image.open(image2)

    # convert the images to numpy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)

    # compare the images using your image similarity model
    similarity_score = image_similarity(image1, image2)

    # return the similarity score as a JSON response
    return render_template("result.html",similarity_score=similarity_score )
    return jsonify({'similarity_score': similarity_score})

# define your image similarity model here
def image_similarity(image1, image2):
    # process the images
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # resize the images to a fixed size
    image1 = cv2.resize(image1, (300, 300))
    image2 = cv2.resize(image2, (300, 300))

    # compute the Structural Similarity Index (SSIM) between the two images
    similarity_score = ssim(image1, image2)

    return similarity_score
if __name__ == '__main__':
    app.run(debug=True)
