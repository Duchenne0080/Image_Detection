from flask import request, Flask, jsonify
import numpy as np
import cv2
import predict
app = Flask(__name__)
@app.route('/detect', methods=['POST'])
def getApplicationsImg():
    getData = request.files.get('image')
    img_data = getData.read()
    np_ary = np.fromstring(img_data, np.uint8)
    cv_image = cv2.imdecode(np_ary, cv2.IMREAD_COLOR)
    predict_Ary = predict.predict_img(cv_image)
    return jsonify(predict_Ary), 201
if __name__ == '__main__':
    app.run(debug=True, host='127.1.1.1', port=8888, threaded=True)
