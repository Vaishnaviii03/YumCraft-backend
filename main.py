from flask import Flask
from flask_cors import CORS
from route import api

app = Flask(__name__)
CORS(app)  # Enable CORS

# Register blueprint from route.py
app.register_blueprint(api)

# Optional: Handle root route to prevent 404 spam
@app.route('/')
def home():
    return 'Yumcraft API is running.', 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
