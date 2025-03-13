from app.app import app, initialize_app
from waitress import serve

initialize_app()

if __name__ == "__main__":
    print("Starting server on http://localhost:8000")
    serve(app, host='0.0.0.0', port=8000) 