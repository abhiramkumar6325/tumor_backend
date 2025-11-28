import os
import json
import numpy as np
import tensorflow as tf

# ------------ Correct Paths Based on Your Structure -------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # fl_server/
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

GLOBAL_MODEL_PATH = os.path.join(STORAGE_DIR, "global_model.h5")
GLOBAL_NPZ_PATH = os.path.join(STORAGE_DIR, "global_flat.npz")
SHAPES_PATH = os.path.join(STORAGE_DIR, "shapes.json")


# -------------------------------------------------------
# Load Global Model
# -------------------------------------------------------
def load_global_model():
    if not os.path.exists(GLOBAL_MODEL_PATH):
        raise FileNotFoundError(f"Global model not found at {GLOBAL_MODEL_PATH}")

    print(f"📌 Loading global model from: {GLOBAL_MODEL_PATH}")
    return tf.keras.models.load_model(GLOBAL_MODEL_PATH)


# -------------------------------------------------------
# Flatten model weights
# -------------------------------------------------------
def get_flattened_weights(model):
    weights = model.get_weights()

    flat = np.concatenate([w.flatten() for w in weights])
    shapes = [w.shape for w in weights]

    return flat, shapes


# -------------------------------------------------------
# Save flattened weights and shapes
# -------------------------------------------------------
def save_global_flat_weights(flat, shapes):
    np.savez(GLOBAL_NPZ_PATH, weights=flat)

    with open(SHAPES_PATH, "w") as f:
        json.dump(shapes, f)


# -------------------------------------------------------
# Convert flat vector back into weight tensors
# -------------------------------------------------------
def rebuild_weights(flat, shapes):
    new_weights = []
    index = 0

    for shape in shapes:
        size = np.prod(shape)
        new_weights.append(flat[index:index + size].reshape(shape))
        index += size

    return new_weights


# -------------------------------------------------------
# Save updated global model
# -------------------------------------------------------
def save_updated_global_model(model):
    print(f"💾 Saving updated global model to: {GLOBAL_MODEL_PATH}")
    model.save(GLOBAL_MODEL_PATH)
