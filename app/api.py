from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from command_recognition_service import CommandRecognitionService


app = Flask(__name__)
api = Api(app)


class PredictCommand(Resource):
    def post(self):
        """Endpoint to predict command
        :return (json): This endpoint returns a json file with the following format:
            {
                "command": "predicted_command"
            }
        """
        # get file from POST request and save it
        audio_file = request.files["file"]

        # get prediction
        service = CommandRecognitionService()
        predicted_command = service.predict(audio_file)

        # send result as a json file
        result = {"command": predicted_command}
        return jsonify(result)


# Endpoints
api.add_resource(PredictCommand, "/predict-command")

if __name__ == "__main__":
    app.run(debug=False)
