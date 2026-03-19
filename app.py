import os
from io import BytesIO
from PIL import Image
import numpy as np
import gradio as gr

from model_stub import run_model_if_available


def upsample_bicubic(img: Image.Image, scale: int = 4) -> Image.Image:
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.BICUBIC)


def process(image, method, scale):
    if image is None:
        return None
    pil = Image.fromarray(image)
    if method == 'bicubic':
        out = upsample_bicubic(pil, scale)
        return np.array(out)
    elif method == 'model':
        out = run_model_if_available(pil, scale)
        if out is None:
            return np.array(pil)
        return np.array(out)
    else:
        return np.array(pil)


def main():
    demo = gr.Interface(
        fn=process,
        inputs=[
            gr.Image(type="numpy", label="Input Image"),
            gr.Radio(['bicubic', 'model'], label='Method', value='bicubic'),
            gr.Slider(2, 8, value=4, step=1, label='Scale')
        ],
        outputs=gr.Image(type="numpy", label="Output Image"),
        title="Super-Resolution Demo",
        description="Baseline bicubic upsampling with an optional PyTorch model if `models/model.pt` is present. See README for GPU optimization steps.",
        allow_flagging='never'
    )
    demo.launch()


if __name__ == '__main__':
    main()
