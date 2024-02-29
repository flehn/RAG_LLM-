from flask import Flask, request, jsonify, redirect
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import RAG


app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)



class Test(Resource):
    def get(self):
        return 'Welcome to, Test App API!'

    def post(self):
        try:
            value = request.get_json()
            if(value):
                
                return jsonify({"message": value['articles']})
            
            return jsonify({"error":"Invalid format."})
            

        except Exception as error:
            return jsonify({"error":error})


class GetPredictionOutput(Resource):
    def get(self):
        return jsonify({"error":"Invalid Method."})
        

    def post(self):
        try:
            data = request.get_json()
            #get context based on article links
            print("enter RAG.read_articles(data['articles'])")
            context_db = RAG.read_articles(data['articles'])
            print("RAG.predict(data['query'], context_db")
            #make a prediction based on the input query and take context into account
            predict = RAG.predict(data['query'], context_db) 
            predictOutput = predict
            return jsonify({"success, data:":data})

        except Exception as error:
            return jsonify({"error":error})


api.add_resource(Test,'/')
api.add_resource(GetPredictionOutput,'/getPredictionOutput')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
