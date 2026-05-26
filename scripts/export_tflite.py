"""Export ViSignRe.keras to ViSignRe.tflite (CPU, SELECT_TF_OPS)."""

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf


def export_tflite(keras_path: str, tflite_path: str) -> None:
    if not os.path.isfile(keras_path):
        raise FileNotFoundError(f"Model not found: {keras_path}")

    print("[INFO] TensorFlow devices:", tf.config.list_physical_devices())
    print("[INFO] Loading Keras model")
    model = tf.keras.models.load_model(keras_path, compile=False)

    print("[INFO] Building TFLite converter")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.experimental_enable_resource_variables = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("[INFO] Converting")
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_path) or ".", exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"[OK] Saved {tflite_path} ({size_mb:.2f} MB)")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_in = os.path.join(root, "models", "ViSignRe.keras")
    default_out = os.path.join(root, "models", "ViSignRe.tflite")

    parser = argparse.ArgumentParser(description="Export ViSignRe Keras to TFLite")
    parser.add_argument("--input", "-i", default=default_in)
    parser.add_argument("--output", "-o", default=default_out)
    args = parser.parse_args()

    try:
        export_tflite(args.input, args.output)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
