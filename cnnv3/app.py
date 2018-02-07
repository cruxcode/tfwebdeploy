from flask import Flask, render_template, request, make_response
from src.model import graph

app = Flask(__name__)

m = graph.model()

@app.route("/", methods = ["GET", "POST"])
def HomePage():
    if request.method == "GET":
        return _send_home_page()
    if request.method == "POST":
        return _process_image()

def _send_home_page():
    return make_response(render_template('index.html'))

def _process_image():
    f = request.files['image']
    pred, prob = m.get_label(f, [299, 299])
    return "Predicted class is " + pred + " with score " + str(prob)

if __name__ == "__main__":
    m.load_graph()
    m.start_session()
    app.run()