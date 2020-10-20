from flask import Flask
from flask import request
from flask import jsonify
import code_flask
import sys
#sys.path.insert(0, '/root/gp/recognition/src')
#import preprocess2
#import new_test_blstm
app = Flask(__name__)

@app.route('/predict')
def get():
    print(request.form)
    print("THAT AINT THE METHOD")
    return 'Shit I SHOULD RETURN'

@app.route('/predict', methods=['POST'])
def post():
   # if 'message' in request.args:
   #     return 'YUP!'
   if 'title' in request.json:
       job_title = request.json['title']
       result = code_flask.final(job_title)
       return result
   #     return jsonify({'task': 'Hello!'})
   #     print(request.json['title'])
    #content = request.data.decode("utf-8")
    #result=preprocess2.process(content)
    #answer=new_test_blstm.test_model(result)
    #return answer
app.run(host='0.0.0.0', port=8000)