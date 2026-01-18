#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import cv2

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Export + OpenCV Test")
    parser.add_argument("--weights", default=None,
                        help="Path to YOLOv8 .pt model (if not provided, downloads yolov8n)")
    parser.add_argument("--variant", default="n",
                        choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Ultralytics not installed. Run: pip install ultralytics onnx onnxruntime onnxsim")

    if args.weights is None:
        model_name = f"yolov8{args.variant}.pt"
        print(f"[INFO] Downloading pretrained {model_name} ...")
        model = YOLO(model_name)
    else:
        model_path = Path(args.weights)
        if not model_path.exists():
            raise FileNotFoundError(f"Weights file {model_path} not found!")
        print(f"[INFO] Using custom weights from {model_path}")
        model = YOLO(model_path)

    output_path = Path(args.output if args.output else f"yolov8{args.variant}.onnx")
    print(f"[INFO] Exporting to ONNX -> {output_path}")
    model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=False, dynamic=False)

    print(f"[INFO] Simplifying {output_path} ...")
    import onnx
    import onnxsim

    model_simp, check = onnxsim.simplify(str(output_path))
    if check:
        onnx.save(model_simp, str(output_path))
        print(f"[DONE] Simplified ONNX saved to {output_path}")
    else:
        print("[WARN] Simplification failed, using original ONNX")

    try:
        onnx_model = onnx.load(output_path)
        print("\n[INFO] ONNX Model Inputs:")
        for inp in onnx_model.graph.input:
            dims = [d.dim_value if d.dim_value > 0 else "dynamic" for d in inp.type.tensor_type.shape.dim]
            print(f"  - {inp.name}: {dims}")

        print("\n[INFO] ONNX Model Outputs:")
        for out in onnx_model.graph.output:
            dims = [d.dim_value if d.dim_value > 0 else "dynamic" for d in out.type.tensor_type.shape.dim]
            print(f"  - {out.name}: {dims}")
    except ImportError:
        print("[WARN] onnx not installed. Skipping model inspection.")

    print("\n[INFO] Testing ONNX model in OpenCV...")
    net = cv2.dnn.readNetFromONNX(str(output_path))
    dummy_img = np.random.randint(0, 256, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(dummy_img, 1/255.0, (args.imgsz, args.imgsz), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    print(f"[INFO] OpenCV ONNX inference output shape: {outputs.shape}")
    print("[DONE] Export + OpenCV test complete!")

if __name__ == "__main__":
    main()
