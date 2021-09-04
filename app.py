from flask import Flask, render_template,request
import pickle
from PIL import Image
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
std_slr=pickle.load(open('std_scaler.pkl','rb'))

lbl={
    0:"Bowen's disease",
    1:'basal cell carcinoma',
    2:'benign keratosis-like lesions',
    3:'dermatofibroma',
    4:'melanoma',
    5:'melanocytic nevi',
    6:'vascular lesions'
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile=request.files['imagefile']
    image_path="./images/"+imagefile.filename
    imagefile.save(image_path)
    
    img=Image.open(image_path)
    img=img.resize((28,28),Image.NEAREST)
    img_arr=np.array(img)
    img_arr=img_arr.reshape(1,2352)
    img_arr=std_slr.transform(img_arr)

    pred=model.predict(img_arr)
    pred=lbl.get(pred[0])

    return render_template('index.html',prediction=pred)

if __name__=="__main__":
    app.run(debug=True)