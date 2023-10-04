from flask import Flask, render_template,request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app=Flask(__name__)

model=load_model(r"model/apple2.h5")

def model_predict(img_path,model):
    test_img=image.load_img(img_path,target_size=(224,224))
    test_img=image.img_to_array(test_img)
    test_img=test_img/255
    test_img=np.expand_dims(test_img,axis=0)
    result=model.predict(test_img)
    return result
@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['file']
        
        basepath=os.path.dirname(os.path.realpath('__file__'))
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        
        result=model_predict(file_path,model)
        
        categories=['Healthy','Multiple Disease','Rust','Scab']
        
        pred_class=result.argmax()
        output=categories[pred_class]
        return output
    return None
if __name__=='__main__':
    app.run(debug=False,port=5926)