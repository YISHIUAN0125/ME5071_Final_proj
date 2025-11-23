import cv2
import numpy as np

def fourier_augmentation(img_src, img_trg, beta=0.01):
    """
    img_src, img_trg: Numpy arrays (H, W, 3) RGB
    beta: 替換中心區域的大小 (0 ~ 1)
    """
    # 確保大小一致
    img_trg = cv2.resize(img_trg, (img_src.shape[1], img_src.shape[0]))

    # FFT
    fft_src = np.fft.fft2(img_src, axes=(0, 1))
    fft_trg = np.fft.fft2(img_trg, axes=(0, 1))

    # Shift (將低頻移到中心)
    fft_src_shift = np.fft.fftshift(fft_src, axes=(0, 1))
    fft_trg_shift = np.fft.fftshift(fft_trg, axes=(0, 1))

    # 取 Amplitude, Phase
    amp_src, pha_src = np.abs(fft_src_shift), np.angle(fft_src_shift)
    amp_trg = np.abs(fft_trg_shift)

    # 替換低頻區域 (風格)
    h, w, _ = img_src.shape
    b_h = int(h * beta)
    b_w = int(w * beta)
    
    c_h, c_w = h // 2, w // 2
    
    # 核心區域替換
    amp_src[c_h-b_h:c_h+b_h, c_w-b_w:c_w+b_w] = amp_trg[c_h-b_h:c_h+b_h, c_w-b_w:c_w+b_w]

    # 反變換
    fft_src_new = amp_src * np.exp(1j * pha_src)
    fft_src_new = np.fft.ifftshift(fft_src_new, axes=(0, 1))
    img_new = np.fft.ifft2(fft_src_new, axes=(0, 1))
    
    img_new = np.abs(img_new)
    img_new = np.clip(img_new, 0, 255).astype(np.uint8)
    
    return img_new