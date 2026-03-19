import os
from PIL import Image
import numpy as np

MODEL_PATH = os.path.join('models', 'model.pt')


def load_model():
    # Placeholder loader: replace with your model load logic (torch.load / torch.jit.load)
    try:
        import torch
        if os.path.exists(MODEL_PATH):
            model = torch.jit.load(MODEL_PATH) if MODEL_PATH.endswith('.jit') else torch.load(MODEL_PATH)
            model.eval()
            return model
    except Exception:
        return None
    return None


def run_model_if_available(pil_img: Image.Image, scale: int = 4):
    model = load_model()
    if model is None:
        # No model available — return None so caller can fallback
        return None
    try:
        import torch
        # Example preprocessing — adapt to your model
        img = np.array(pil_img).astype('float32') / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
        out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        out_img = Image.fromarray((out * 255).astype('uint8'))
        return out_img
    except Exception:
        return None
