#set up flask on replit: https://replit.com/talk/learn/Flask-Tutorial-Part-1-the-basics/26272
#setting up a model to a web app: https://blog.cambridgespark.com/deploying-a-machine-learning-model-to-the-web-725688b851c7

import flask
import pickle

#use pickle to open up the cached model; this doesn't work as of now and we need to fix it

with open(f'model/stock_predictor_model_a.pkl', 'rb') as f:
	model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
	if flask.request.method == 'GET':
		return(flask.render_template('main.html'))
	if flask.request.method == 'POST':
		'''
		symbol = flask.request.form['symbol']
    	start_date = flask.request.form['humidity']
		end_date = flask.request.form['windspeed']
		input_variables = pd.DataFrame([[symbol, start_date, end_date]],columns=['temperature', 'humidity', 'windspeed'], dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html', original_input={'Temperature':temperature,'Humidity':humidity,'Windspeed':windspeed},result=prediction,)
		'''

if __name__ == '__main__':
	app.run(host="0.0.0.0", port="8888")