import os

from flask import Flask, request, render_template, send_file
from datetime import datetime

from musti_image import MustiImage
from musti_model import MustiModel

app = Flask(__name__)

image_name = ''

"""http://musti.be?time=18-04-2022T22:00:00"""
@app.route('/', methods=['GET'])
def main():  # put application's code here
    """get the parameters"""
    args = request.args
    time = args.get('time')
    if time is not None:
        now_obj = datetime.strptime(time, '%d-%m-%YT%H%M%S')
        # time, load corresponding image
    else:
        now_obj = datetime.now()
        # get current time and load corresponding image

    musti_image_object = MustiImage()
    musti_model_object = MustiModel()

    """get the musti image
       load the model
       predict"""
    bad_image = True
    offset = 0

    while bad_image:
        musti_candidate = musti_image_object.load_musti_image_for_datetime(now_obj, offset)
        musti_preprocessed = musti_image_object.preprocess_image(musti_candidate)
        musti_classified = musti_model_object.model.predict(musti_preprocessed)
        # ToDo: implement accuracy
        if musti_classified[0] == 1:
            classification_result = "BINNEN"
            bad_image = False
        elif musti_classified[0] == 2:
            classification_result = "BUITEN"
            bad_image = False
        else:
            offset = musti_image_object.getOffset()
        classification_name = musti_image_object.getImageName()
        image_name = classification_name

    return render_template('main.html', classification_result=classification_result, classification_name=classification_name, image_name=image_name)

