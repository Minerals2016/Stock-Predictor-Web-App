# set up flask on replit: https://replit.com/talk/learn/Flask-Tutorial-Part-1-the-basics/26272
# setting up a model to a web app: https://blog.cambridgespark.com/deploying-a-machine-learning-model-to-the-web-725688b851c7

import flask
import pickle
from model import model_script

# use pickle to open up the cached model; this doesn't work as of now and we need to fix it

"""with open(f'model/stock_predictor_model_a.pkl', 'rb') as f:
	model = pickle.load(f)"""

"""symbol = input("What stock would you like to analyze? ")
symbol = symbol.upper()
start = input("Start date (yyyy-mm-dd): ")
end = input("End date (yyyy-mm-dd): ")"""

symbol = "AAPL"
start = "2012-01-01"
end = "2019-12-17"

# model_script.model(symbol, start, end)

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    # model_script.model()

    if flask.request.method == 'POST':
        symbol = flask.request.form['temperature']
        start_date = flask.request.form['humidity']
        end_date = flask.request.form['windspeed']

        return flask.render_template('main.html', original_input={'Temperature': symbol, 'Humidity': start_date,
                                                                  'Windspeed': end_date}, result={}}, )

        if __name__ == '__main__':
            app.run(host="0.0.0.0", port="8888")
