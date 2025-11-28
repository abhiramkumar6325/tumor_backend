import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

MODEL_PATH = os.path.join(STORAGE_DIR, "global_model.h5")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

    print(f"📌 Loading model from: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)
