import os
import pickle
import tensorflow as tf

# Optional: confirm version
print("TensorFlow version:", tf.__version__)

# ✅ Step 1: Define paths
archive_path = os.path.join("Archive", "baseline.h5")
scaler_path = os.path.join("Archive", "baseline_scalers.pkl")

# ✅ Step 2: Load the model
model = tf.keras.models.load_model(archive_path, compile=False)

# ✅ Step 3: Load the scalers
with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]

print("✅ Model and scalers loaded successfully.")
