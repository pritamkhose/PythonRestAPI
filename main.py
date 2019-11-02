from app import app, mongo
from bson.json_util import dumps
from bson.objectid import ObjectId
from flask import jsonify, flash, request, make_response
from werkzeug import generate_password_hash, check_password_hash
import traceback
from pprint import pprint

# https://api.mongodb.com/python/current/api/pymongo/results.html
# https://stackoverflow.com/questions/11773348/python-flask-how-to-set-content-type
# https://stackoverflow.com/questions/4564559/get-exception-description-and-stack-trace-which-caused-an-exception-all-as-a-st
# https://stackoverflow.com/questions/49355010/how-do-i-watch-python-source-code-files-and-restart-when-i-save
# run code --> python main.py or nodemon main.py

import datetime
import time
import sys
import os
import shutil

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import joblib
import pandas as pd


@app.route('/add', methods=['POST'])
def add_user():
    try:
        _json = request.json
        _name = _json['name']
        _email = _json['email']
        _password = _json['pwd']
        # validate the received values
        if _name and _email and _password and request.method == 'POST':
            # do not save password as a plain text
            _hashed_password = generate_password_hash(_password)
            # add details
            op = mongo.db.user.insert(
                {'name': _name, 'email': _email, 'pwd': _hashed_password})
            response = make_response(jsonify({'result': str(op)}))
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            response.status_code = 201
        else:
            return not_found()
    except Exception as e:
        response = make_response(jsonify({
            'status': 500,
            'message': 'Exception occurred',
            'exception': str(traceback.format_exc())
        }))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        response.status_code = 500
    # response = jsonify('User added successfully!') #, mimetype='application/json'
    # response.status_code = 200
    return response


@app.route('/users')
def users():
    users = mongo.db.user.find()
    resp = dumps(users)
    # return jsonify(resp)
    response = make_response(resp)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/user/<id>')
def user(id):
    user = mongo.db.user.find_one({'_id': ObjectId(id)})
    # response = dumps(user)
    response = make_response(dumps(user))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/update', methods=['PUT'])
def update_user():
    try:
        _json = request.json
        _id = _json['_id']
        _name = _json['name']
        _email = _json['email']
        _password = _json['pwd']
        # validate the received values
        if _name and _email and _password and _id and request.method == 'PUT':
            # do not save password as a plain text
            _hashed_password = generate_password_hash(_password)
            # save edits
            op = mongo.db.user.update_one({'_id': ObjectId(_id['$oid']) if '$oid' in _id else ObjectId(
                _id)}, {'$set': {'name': _name, 'email': _email, 'pwd': _hashed_password}})
            response = make_response(jsonify({'result': op.matched_count}))
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
        else:
            return not_found()
    except Exception as e:
        response = make_response(jsonify({
            'status': 500,
            'message': 'Exception occurred',
            'exception': str(traceback.format_exc())
        }))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        response.status_code = 500
    return response


@app.route('/delete/<id>', methods=['DELETE'])
def delete_user(id):
    op = mongo.db.user.delete_one({'_id': ObjectId(id)})
    response = make_response(jsonify({'result': op.deleted_count}))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/')
def DateTimeMethod():
    resp = {'Message': 'Server is running!',
            'date': datetime.datetime.now(), 'URL': 'http://localhost:5000/'}
    response = make_response(resp)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/savemodel')
def savemodel():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=2018)

    # Load the model from the file
    # knn_from_joblib = joblib.load('knn_model.pkl')
    # Use the loaded model to make predictions
    # resp =  knn_from_joblib.predict(X_test)
    response = make_response({'X_test': str(X_test)})  # , 'resp': str(resp)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


# https://github.com/amirziai/sklearnflask
# inputs
training_data = 'data/titanic.csv'
include = ['Age', 'Sex', 'Embarked', 'Survived']
dependent_variable = include[-1]
model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory
# These will be populated at training time
model_columns = None
clf = None


@app.route('/train', methods=['GET'])
def train():
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # using random forest as an example
    # can do the training separately and just update the pickles
    from sklearn.ensemble import RandomForestClassifier as rf

    df = pd.read_csv(training_data)
    df_ = df[include]
    categoricals = []  # going to one-hot encode categorical variables
    for col, col_type in df_.dtypes.items():
        if col_type == 'O':
            categoricals.append(col)
        else:
            # fill NA's with 0 for ints/floats, too generic
            df_[col].fillna(0, inplace=True)

    # get_dummies effectively creates one-hot encoded variables
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(x.columns)
    joblib.dump(model_columns, model_columns_file_name)

    global clf
    clf = rf()
    start = time.time()
    clf.fit(x, y)

    joblib.dump(clf, model_file_name)

    response = make_response({
        'Message': 'Success', 'Trained in seconds': (time.time() - start),
        'Model training score': clf.score(x, y),
        'max_features': str(clf.max_features),
    })
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree('model')
        response = make_response({'Message': 'Model wiped'})
    except Exception as e:
        response = make_response(
            {'Message': 'Could not remove and recreate the model directory', 'Exception': str(e)})
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    response.status_code = 500
    return response


@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))

            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(clf.predict(query))

            # Converting to int from int64
            return jsonify({"prediction": list(map(int, prediction)), "input": json_})

        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        response = make_response({'Message': 'train model first'})
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        response.status_code = 400
        return response

# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
@app.route('/cm', methods=['POST'])
def cmcrac():
    try:
        # x = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0] y = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
        json_ = request.json
        x = json_['x']
        y = json_['y']

        from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
        cm = confusion_matrix(x, y)
        cr = classification_report(x, y)
        ac = accuracy_score(x, y)
        response = make_response(
            {'result': 'ok', 'confusion_matrix': str(cm), 'classification_report': str(cr), 'accuracy_score': str(ac), })
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    except Exception as e:
        return jsonify({'result': 'error', 'error': str(e), 'trace': traceback.format_exc()})

# https://youtu.be/KfnhNlD8WZI
# https://www.geeksforgeeks.org/saving-a-machine-learning-model/
@app.route('/trainknn', methods=['GET'])
def trainknn():
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier as KNN
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=2018)
    # import train predict KNeighborsClassifier
    knn = KNN(n_neighbors=3)
    knn.fit(X_train, y_train)
    knn_predict = knn.predict(X_test)

    # Save the model as a pickle in a file
    joblib.dump(knn, './model/knn_model.pkl')
    # Load the model from the file
    knn_from_joblib = joblib.load('./model/knn_model.pkl')
    knn_from_joblib_predict = knn_from_joblib.predict(X_test)
    
    response = make_response({
        'Message': 'ok',
        'X': str(X), 'y': str(y),
        'X_train': str(X_train), 'X_test': str(X_test), 'y_train': str(y_train), 'y_test': str(y_test),
        'knn_predict': str(knn_predict), 'knn_from_joblib_predict': str(knn_from_joblib_predict),
        'model score': knn_from_joblib.score(X, y),
        'model score train': knn_from_joblib.score(X_train, y_train),
        'model score test': knn_from_joblib.score(X_test, y_test),
    })
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    response.status_code = 400
    return response


@app.errorhandler(404)
def not_found(error=None):
    response = make_response(jsonify({
        'status': 404,
        'message': 'Not Found: ' + request.url,
    }))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    response.status_code = 404
    return response


if __name__ == "__main__":
    app.run(port=8080)
    # app.run(debug=False, host="0.0.0.0", threaded=False)
