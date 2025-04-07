from flask import Flask
from flask_cors import CORS
from route import api

app = Flask(__name__)
CORS(app)  # Enable CORS

# Register blueprint from route.py
app.register_blueprint(api)

if __name__ == '__main__':
    app.run(debug=True)
