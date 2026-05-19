import numpy as np
from PIL import Image
import os

def build_dct_matrix(N=8):
    D = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        for x in range(N):
            if k == 0:
                D[k, x] = 1.0 / np.sqrt(N)
            else:
                D[k, x] = np.sqrt(2.0/N) * np.cos(np.pi*k*(2*x+1)/(2*N))
    return D

DCT_MATRIX   = build_dct_matrix(8)
DCT_MATRIX_T = DCT_MATRIX.T

def dct2d(block):
    """2D DCT: F = D @ block @ D.T"""
    return DCT_MATRIX @ block @ DCT_MATRIX_T

def idct2d(block):
    """2D IDCT: f = D.T @ F @ D"""
    return DCT_MATRIX_T @ block @ DCT_MATRIX

def dct2d_batch(blocks):
    """batch DCT untuk shape (nh, nw, 8, 8)"""
    return DCT_MATRIX @ blocks @ DCT_MATRIX_T

def idct2d_batch(blocks):
    return DCT_MATRIX_T @ blocks @ DCT_MATRIX

ZIGZAG_8x8 = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]
WM_COEF_INDICES = list(range(10, 18))
WM_COORDS = [ZIGZAG_8x8[i] for i in WM_COEF_INDICES]

def generate_watermark_sequence(text, length, seed=42):
    np.random.seed(seed + sum(ord(c) for c in text))
    pattern = np.random.choice([-1.0, 1.0], size=length)
    bits = []
    for char in text:
        val = ord(char)
        for i in range(7, -1, -1):
            bits.append(1.0 if (val >> i) & 1 else -1.0)
    full_bits = np.array((bits * (length // len(bits) + 1))[:length])
    return pattern * full_bits

def image_to_blocks(channel, B=8):
    H, W = channel.shape
    nh, nw = H//B, W//B
    t = channel[:nh*B, :nw*B]
    return t.reshape(nh, B, nw, B).transpose(0,2,1,3), nh, nw

def blocks_to_image(blocks, nh, nw, B=8):
    return blocks.transpose(0,2,1,3).reshape(nh*B, nw*B)

def embed_dct(image_path, watermark_text, output_path, alpha=10.0, block_size=8):
    img = Image.open(image_path).convert("YCbCr")
    arr = np.array(img, dtype=np.float64)
    Y, Cb, Cr = arr[:,:,0].copy(), arr[:,:,1].copy(), arr[:,:,2].copy()
    H_orig, W_orig = Y.shape

    blocks, nh, nw = image_to_blocks(Y, block_size)
    total_cap = nh * nw * len(WM_COEF_INDICES)
    print(f"[DCT] {H_orig}x{W_orig}, blok {nh}x{nw}={nh*nw}, kapasitas={total_cap} bit, alpha={alpha}")

    wm_seq  = generate_watermark_sequence(watermark_text, total_cap)
    wm_grid = wm_seq.reshape(nh, nw, len(WM_COEF_INDICES))

    dct_b = dct2d_batch(blocks)
    for k, (r, c) in enumerate(WM_COORDS):
        dct_b[:,:,r,c] += alpha * wm_grid[:,:,k]
    wm_blocks = idct2d_batch(dct_b)

    Y_wm = Y.copy()
    Y_wm[:nh*block_size, :nw*block_size] = np.clip(blocks_to_image(wm_blocks, nh, nw, block_size), 0, 255)

    out = np.stack([Y_wm, Cb, Cr], axis=2).astype(np.uint8)
    Image.fromarray(out, mode="YCbCr").convert("RGB").save(output_path)
    print(f"[DCT] Tersimpan: {output_path}")

def extract_dct(original_path, watermarked_path, watermark_text, alpha=10.0, block_size=8):
    Y_o = np.array(Image.open(original_path).convert("YCbCr"), dtype=np.float64)[:,:,0]
    Y_w = np.array(Image.open(watermarked_path).convert("YCbCr"), dtype=np.float64)[:,:,0]

    bo, nh, nw = image_to_blocks(Y_o, block_size)
    bw, _,  _  = image_to_blocks(Y_w, block_size)

    do = dct2d_batch(bo)
    dw = dct2d_batch(bw)

    total_cap = nh * nw * len(WM_COEF_INDICES)
    expected  = generate_watermark_sequence(watermark_text, total_cap).reshape(nh, nw, len(WM_COEF_INDICES))

    ext, exp = [], []
    for k, (r, c) in enumerate(WM_COORDS):
        ext.extend((dw[:,:,r,c] - do[:,:,r,c]).flatten())
        exp.extend(expected[:,:,k].flatten())

    ext, exp = np.array(ext), np.array(exp)
    ne, nx = np.linalg.norm(ext), np.linalg.norm(exp)
    if ne == 0 or nx == 0:
        return 0.0
    return float(np.dot(ext, exp) / (ne * nx))

def compress_jpeg(input_path, output_path, quality):
    Image.open(input_path).convert("RGB").save(output_path, format="JPEG", quality=quality, optimize=True)
    return os.path.getsize(output_path)

def calculate_psnr(a_path, b_path):
    a = np.array(Image.open(a_path).convert("RGB"), dtype=np.float64)
    b = np.array(Image.open(b_path).convert("RGB"), dtype=np.float64)
    mse = np.mean((a-b)**2)
    return float('inf') if mse == 0 else round(10*np.log10(255**2/mse), 4)

def evaluate_dct_robustness(original_path, watermarked_png, watermark_text, alpha, qf_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    THRESHOLD = 0.3
    results = {}
    print(f"\n{'QF':>4} | {'File Size':>12} | {'Correlation':>12} | {'Detected':>10}")
    print("-"*55)
    for qf in qf_list:
        cp = os.path.join(output_dir, f"dct_compressed_qf{qf:02d}.jpg")
        fs = compress_jpeg(watermarked_png, cp, qf)
        corr = extract_dct(original_path, cp, watermark_text, alpha)
        det  = corr > THRESHOLD
        results[qf] = {"file_size_bytes": fs, "correlation": round(corr,4), "detected": det}
        print(f"{qf:>4} | {fs:>10} B | {corr:>12.4f} | {'DETECTED' if det else 'NOT FOUND'}")
    return results

if __name__ == "__main__":
    BASE   = os.path.dirname(os.path.abspath(__file__))
    IMG_IN = os.path.join(BASE, "..", "face.jpg")
    WM_PNG = os.path.join(BASE, "dct_watermarked.png")
    OUT_DIR= os.path.join(BASE, "results")
    ALPHA  = 10.0
    WM_TEXT= "Copyright2024-MyName"

    print("="*70)
    print("DCT WATERMARKING — IMPLEMENTASI DARI NOL (VECTORIZED)")
    print("="*70)

    print("\n[STEP 1] Menyisipkan watermark DCT...")
    embed_dct(IMG_IN, WM_TEXT, WM_PNG, alpha=ALPHA)

    print("\n[STEP 2] Verifikasi dari PNG...")
    corr = extract_dct(IMG_IN, WM_PNG, WM_TEXT, ALPHA)
    print(f"  Correlation: {corr:.4f}  | Status: {'BERHASIL' if corr > 0.3 else 'GAGAL'}")

    print("\n[STEP 3] PSNR...")
    print(f"  PSNR = {calculate_psnr(IMG_IN, WM_PNG)} dB")

    print("\n[STEP 4] Evaluasi robustness vs JPEG QF...")
    results = evaluate_dct_robustness(IMG_IN, WM_PNG, WM_TEXT, ALPHA,
                                       [95,90,80,70,60,50,40,30,20,10], OUT_DIR)

    print("\n[STEP 5] RINGKASAN:")
    for qf, r in sorted(results.items(), reverse=True):
        print(f"  QF {qf:>2}: {'DETECTED' if r['detected'] else 'NOT FOUND'}  (corr={r['correlation']:.4f})")
