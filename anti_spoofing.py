import os
import cv2
import numpy as np
import logging
import onnxruntime as ort
from config import LIVENESS_THRESHOLD

log = logging.getLogger("System")

class AntiSpoofing:
    """
    A robust anti-spoofing mechanism using MiniFASNet (Silent-Face-Anti-Spoofing).
    This deep learning model analyzes face texture, lighting, and moire patterns 
    to mathematically distinguish between a real 3D face and a 2D screen/photo.
    """
    def __init__(self):
        model_path = os.path.join("data", "models", "minifasnet.onnx")
        self.session = None
        
        try:
            if not os.path.exists(model_path):
                log.error(f"[SPOOF] MiniFASNet model not found at {model_path}!")
                return
                
            # Try to use GPU if available, otherwise fallback to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            log.info("[SPOOF] MiniFASNet ONNX Anti-spoofing model loaded successfully.")
            self.input_name = self.session.get_inputs()[0].name
        except Exception as e:
            log.error(f"[SPOOF] Failed to load MiniFASNet ONNX: {e}")

    def available(self):
        return self.session is not None

    def check_liveness(self, frame, bbox):
        """
        Check if the face is real using MiniFASNet.
        Returns: (is_real: bool, liveness_score: float)
        """
        if self.session is None:
            return True, 1.0
            
        fx1, fy1, fx2, fy2 = bbox
        
        # MiniFASNet requires a slightly expanded crop (usually 2.7x the face bbox)
        # to capture the surrounding context (bezel of phone, edges of paper).
        w = fx2 - fx1
        h = fy2 - fy1
        
        if w <= 0 or h <= 0:
            return True, 1.0
            
        # Calculate center
        cx = fx1 + w / 2.0
        cy = fy1 + h / 2.0
        
        # Expand scale
        scale = 2.7
        new_w = w * scale
        new_h = h * scale
        
        # New bbox
        nx1 = int(cx - new_w / 2.0)
        ny1 = int(cy - new_h / 2.0)
        nx2 = int(cx + new_w / 2.0)
        ny2 = int(cy + new_h / 2.0)
        
        # Constrain to frame
        frame_h, frame_w = frame.shape[:2]
        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(frame_w, nx2)
        ny2 = min(frame_h, ny2)
        
        # Crop
        face_crop = frame[ny1:ny2, nx1:nx2]
        if face_crop.size == 0:
            return True, 1.0
            
        # Preprocess for MiniFASNet (80x80, BGR, [0, 1] range, NCHW format)
        try:
            resized_crop = cv2.resize(face_crop, (80, 80))
            input_tensor = resized_crop.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1)) # HWC to CHW
            input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dim NCHW
            
            # Run inference
            output = self.session.run(None, {self.input_name: input_tensor})[0]
            
            # Softmax to get probabilities
            exp_out = np.exp(output - np.max(output)) # numeric stability
            probs = exp_out / np.sum(exp_out)
            
            # MiniFASNet classes: 0 = Print (Paper), 1 = Real (Live), 2 = Replay (Screen)
            # We extract class 1 as the definitive Liveness Score.
            num_classes = probs.shape[1]
            if num_classes == 3:
                liveness_score = float(probs[0][1])
            elif num_classes == 2:
                # If a custom 2-class model: usually 1 is Real
                liveness_score = float(probs[0][1])
            else:
                return True, 1.0
                
            is_real = liveness_score >= LIVENESS_THRESHOLD
            
            if not is_real:
                log.warning(f"[SPOOF] MiniFASNet rejected face! Liveness Score: {liveness_score:.3f} < {LIVENESS_THRESHOLD}")
                
            return is_real, liveness_score
            
        except Exception as e:
            log.error(f"[SPOOF] Inference error: {e}")
            return True, 1.0
