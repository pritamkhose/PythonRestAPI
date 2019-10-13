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
    app.run()
