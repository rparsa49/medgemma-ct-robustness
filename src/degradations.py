"""Deterministic, severity-ordered perturbations of display-windowed CT images."""
from io import BytesIO
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import convolve


def degrade(image: Image.Image, kind: str, severity: int, seed: int) -> Image.Image:
    if severity not in range(1, 6):
        raise ValueError('Severity must be 1..5')
    rng = np.random.default_rng(seed)
    image = image.convert('RGB')
    arr = np.asarray(image).astype(np.float32) / 255.0
    if kind == 'gaussian_noise':
        arr = np.clip(arr + rng.normal(0, [0.02, .04, .07, .10, .15][severity-1], arr.shape), 0, 1)
    elif kind == 'gaussian_blur':
        return image.filter(ImageFilter.GaussianBlur([.6, 1.2, 2, 3, 5][severity-1]))
    elif kind == 'motion_blur':
        length = [3, 5, 9, 13, 19][severity-1]
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length//2, :] = 1 / length
        arr = np.stack([convolve(arr[..., c], kernel, mode='reflect') for c in range(3)], axis=-1)
    elif kind == 'low_contrast':
        factor = [.8, .6, .45, .3, .15][severity-1]
        arr = np.clip((arr - .5) * factor + .5, 0, 1)
    elif kind == 'resolution_loss':
        factor = [2, 3, 4, 6, 8][severity-1]
        w, h = image.size
        return image.resize((max(1,w//factor),max(1,h//factor)), Image.Resampling.BILINEAR).resize((w,h), Image.Resampling.BILINEAR)
    elif kind == 'jpeg':
        buf = BytesIO()
        image.save(buf, format='JPEG', quality=[65, 40, 25, 12, 5][severity-1])
        buf.seek(0)
        with Image.open(buf) as compressed:
            return compressed.convert('RGB')
    elif kind == 'bias_field':
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
        angle = rng.uniform(0, 2*np.pi)
        field = np.cos(angle)*xx + np.sin(angle)*yy
        amplitude = [.08, .16, .24, .32, .45][severity-1]
        arr = np.clip(arr * (1 + amplitude*field[..., None]), 0, 1)
    else:
        raise ValueError(f'Unknown degradation: {kind}')
    return Image.fromarray(np.uint8(np.round(arr*255)), 'RGB')
