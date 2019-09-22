from flask import Flask
from flask_pymongo import PyMongo

app = Flask(__name__)
app.secret_key = "secret key"
# app.config["MONGO_URI"] = "mongodb://localhost:27017/roytuts"
app.config["MONGO_URI"] = "mongodb://**Username**:**Password**@cluster0-shard-00-00-qhmqk.mongodb.net:27017,cluster0-shard-00-01-qhmqk.mongodb.net:27017,cluster0-shard-00-02-qhmqk.mongodb.net:27017/python?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true"
mongo = PyMongo(app)
