import os
import subprocess
import time
import signal
import sys

def start_mlflow_server():
    """Start the MLflow tracking server"""
    print("Starting MLflow tracking server...")
    
    # Create mlruns directory if it doesn't exist
    os.makedirs("mlruns", exist_ok=True)
    
    # Start MLflow server
    mlflow_process = subprocess.Popen(
        ["mlflow", "server", 
         "--host", "0.0.0.0", 
         "--port", "5001", 
         "--backend-store-uri", "mlruns"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give the server time to start
    time.sleep(2)
    
    # Check if process is running
    if mlflow_process.poll() is None:
        print("MLflow server started successfully on port 5001")
        return mlflow_process
    else:
        stderr = mlflow_process.stderr.read().decode()
        print(f"Failed to start MLflow server: {stderr}")
        return None

def run_model_training():
    """Run MLflow model training"""
    print("Training and logging model to MLflow...")
    
    # Import MLflow integration and run training
    from mlflow_integration import train_and_log_model
    
    # Set tracking URI to our MLflow server
    train_and_log_model(tracking_uri="http://localhost:5001")
    
    print("Model training and logging complete!")

def update_app_with_mlflow():
    """Update the Flask app to use MLflow for model loading"""
    print("Updating Flask app to use MLflow for model management...")
    
    # Create an updated app file with MLflow integration
    with open("app.py", "r") as f:
        app_code = f.read()
    
    # Add MLflow imports
    mlflow_imports = """import mlflow
import mlflow.sklearn
"""
    
    # Check if already added
    if "import mlflow" not in app_code:
        app_code = app_code.replace(
            "import joblib", 
            "import joblib\n" + mlflow_imports
        )
    
    # Update model loading code
    model_loading_code = """
# Load the model - try MLflow first, fall back to local file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'random_forest_model.pkl')
try:
    # Try to load the best model from MLflow
    mlflow.set_tracking_uri("http://localhost:5001")
    from mlflow_integration import load_best_model_from_mlflow
    model = load_best_model_from_mlflow(tracking_uri="http://localhost:5001", metric="r2_test")
    print("Loaded best model from MLflow")
except Exception as e:
    print(f"Error loading model from MLflow: {e}")
    print("Falling back to local model file")
    # Check if model exists, otherwise train it
    if not os.path.exists(MODEL_PATH):
        from model import train_and_save_model
        train_and_save_model(MODEL_PATH)
    model = joblib.load(MODEL_PATH)
"""
    
    # Replace model loading section
    if "MODEL_PATH = os.path.join" in app_code:
        app_code = app_code.replace(
            "# Load the trained model\nMODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'random_forest_model.pkl')\n\n# Check if model exists, otherwise train it\nif not os.path.exists(MODEL_PATH):\n    from model import train_and_save_model\n    train_and_save_model(MODEL_PATH)\n\nmodel = joblib.load(MODEL_PATH)",
            "# Load the trained model" + model_loading_code
        )
    
    # Add MLflow route for accessing the UI
    mlflow_route = """
@app.route('/mlflow')
def redirect_to_mlflow():
    \"\"\"Redirect to MLflow UI\"\"\"
    return redirect("http://localhost:5001")
"""
    
    if "@app.route('/mlflow')" not in app_code:
        # Add import for redirect
        if "from flask import redirect" not in app_code:
            app_code = app_code.replace(
                "from flask import Flask, render_template, request, jsonify",
                "from flask import Flask, render_template, request, jsonify, redirect"
            )
        
        # Add the MLflow route
        app_code = app_code.replace(
            "@app.route('/docs')\ndef api_docs():",
            mlflow_route + "\n@app.route('/docs')\ndef api_docs():"
        )
    
    # Write the updated app code
    with open("app_with_mlflow.py", "w") as f:
        f.write(app_code)
    
    print("Flask app updated with MLflow integration - saved as app_with_mlflow.py")

def install_mlflow():
    """Install MLflow package if not already installed"""
    print("Checking for MLflow installation...")
    try:
        import mlflow
        print("MLflow is already installed.")
    except ImportError:
        print("Installing MLflow...")
        try:
            subprocess.check_call(["pip", "install", "mlflow"])
            print("MLflow installed successfully!")
        except subprocess.CalledProcessError:
            print("Failed to install MLflow. Please install it manually using 'pip install mlflow'")
            sys.exit(1)

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nShutting down...")
        if 'mlflow_process' in locals() and mlflow_process:
            mlflow_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Install MLflow if needed
    install_mlflow()
    
    # Start MLflow server
    mlflow_process = start_mlflow_server()
    
    if mlflow_process:
        # Run model training with MLflow
        run_model_training()
        
        # Update app with MLflow integration
        update_app_with_mlflow()
        
        print("\n" + "="*50)
        print("MLflow setup complete!")
        print("="*50)
        print("\nNext steps:")
        print("1. Access MLflow UI: http://localhost:5001")
        print("2. Run the Flask app with MLflow integration:")
        print("   python app_with_mlflow.py")
        print("\nPress Ctrl+C to shut down the MLflow server.")
        
        # Keep the script running to maintain the MLflow server
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down MLflow server...")
            mlflow_process.terminate()
            print("Done!")
