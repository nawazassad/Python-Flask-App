from flask import Flask, render_template, request
from wtforms import Form, StringField
import joblib
from wtforms.validators import (DataRequired,Length)

#loading the pkl files which were converted to get the prediction model and vector
load_vector= joblib.load("vectorizer.pkl")
load_model= joblib.load("model.pkl")

app = Flask(__name__)

#converting 0 and 1 to negative and positive
def prediction(comment_for_analysis):
 label = {0: 'negative', 1: 'positive'}
 X = load_vector.transform([comment_for_analysis])
 y = load_model.predict(X)[0]
 return label[y]

#Defining the comment section
class CommentForm(Form):
 comment = StringField('',[DataRequired(),Length(min=4,message=('Your comment'))])


#declaring the home route
@app.route('/')
def index():
 form = CommentForm(request.form)
 return render_template('commentform.html', form=form)

#declaring the route for results
@app.route('/results', methods=['POST'])
def results():
 form = CommentForm(request.form)
 if request.method == 'POST' and form.validate():
  post_comment = request.form['comment']
  y = prediction(post_comment)

  return render_template('results.html',content=post_comment,predicted_value=y)
  return render_template('commentform.html', form=form)


if __name__ == '__main__':
 app.run(debug=True)