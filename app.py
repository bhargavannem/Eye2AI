import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from predict import predict
app = Flask(__name__,template_folder='templates')
app.config['IMAGE_UPLOADS'] = "static/uploads"
app.config['ALLOWED_IMAGE_EXTENSIONS'] = ['JPEG', 'JPG']

def allowed_image(filename):
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if image.filename == "":
                print("Image must have have a filename")
                return redirect(request.url)

            if not allowed_image(image.filename):
                print("That image extension is not allowed")
                return redirect(request.url)

            else:
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
            print("Image saved...")
            prd = predict(filename)
            os.remove(os.path.join(app.config["IMAGE_UPLOADS"],filename))
            return render_template('home.html',cnv=prd[0], dme=prd[1], drusen=prd[2], normal=prd[3], prediction=True)
    else:
        return render_template('home.html')

if __name__ == "___main__":
    app.run()