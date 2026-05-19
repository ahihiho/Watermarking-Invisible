import numpy as np
from PIL import Image
import os

# ─────────────────────────────────────────────
# BAGIAN 1: ENCODE — Menyisipkan watermark
# ─────────────────────────────────────────────

def text_to_bits(text: str) -> list[int]:
    """
    Konversi string teks → list bit (0/1).
    Setiap karakter → 8 bit ASCII.
    Tambahkan delimiter '11111110' (0xFE) sebagai penanda akhir pesan.
    """
    bits = []
    for char in text:
        ascii_val = ord(char)           # karakter → angka ASCII
        for i in range(7, -1, -1):      # ambil bit dari MSB ke LSB
            bits.append((ascii_val >> i) & 1)
    # Delimiter: 8 bit '1' diikuti '0' → penanda akhir
    bits.extend([1, 1, 1, 1, 1, 1, 1, 0])
    return bits


def embed_lsb(image_path: str, watermark_text: str, output_path: str) -> np.ndarray:
    """
    Sisipkan watermark ke gambar menggunakan metode LSB.

    Cara kerja:
    - Setiap pixel RGB punya 3 channel, masing-masing 8 bit.
    - Kita ubah bit terakhir (LSB) setiap channel dengan bit watermark.
    - Perubahan 1 bit = perubahan nilai 0 atau 1 pada skala 0–255 → tidak terlihat mata.

    Args:
        image_path    : path gambar asli
        watermark_text: teks yang akan disembunyikan
        output_path   : path output gambar ber-watermark (PNG, lossless)

    Returns:
        numpy array gambar hasil watermark
    """
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img, dtype=np.uint8)   # shape: (H, W, 3)

    bits = text_to_bits(watermark_text)
    total_bits = len(bits)
    H, W, C = pixels.shape

    # Cek kapasitas: maksimal H*W*3 bit bisa disimpan
    capacity = H * W * C
    if total_bits > capacity:
        raise ValueError(f"Watermark terlalu panjang! Kapasitas: {capacity} bit, dibutuhkan: {total_bits} bit")

    # Flatten pixel array agar mudah di-iterate
    flat = pixels.flatten().copy()   # shape: (H*W*3,)

    for idx, bit in enumerate(bits):
        # Metode: clear LSB dengan AND 0xFE (11111110), lalu OR dengan bit watermark
        flat[idx] = (flat[idx] & 0b11111110) | bit

    # Kembalikan ke shape asli
    watermarked = flat.reshape(H, W, C)

    # Simpan sebagai PNG (lossless) dulu untuk preservasi LSB
    Image.fromarray(watermarked).save(output_path)
    print(f"[LSB] Watermark berhasil disimpan ke: {output_path}")
    print(f"[LSB] Bit yang disisipkan: {total_bits} bit ({total_bits//8} karakter + delimiter)")

    return watermarked


# ─────────────────────────────────────────────
# BAGIAN 2: DECODE — Ekstrak watermark
# ─────────────────────────────────────────────

def bits_to_text(bits: list[int]) -> str:
    """
    Konversi list bit → string teks.
    Baca 8 bit sekaligus, konversi ke karakter ASCII.
    Berhenti saat menemukan delimiter (11111110 = 0xFE).
    """
    text = ""
    for i in range(0, len(bits) - 7, 8):
        byte = bits[i:i+8]
        val = 0
        for b in byte:
            val = (val << 1) | b
        if val == 0xFE:      # delimiter ditemukan → stop
            break
        if val == 0:         # null byte → stop (safety)
            break
        text += chr(val)
    return text


def extract_lsb(image_path: str, num_chars_estimate: int = 500) -> str:
    """
    Ekstrak watermark dari gambar ber-watermark.

    Cara kerja:
    - Baca LSB dari setiap nilai channel pixel secara berurutan.
    - Kumpulkan bit-bit tersebut, konversi ke teks.
    - Hentikan saat delimiter ditemukan.

    Args:
        image_path        : path gambar ber-watermark
        num_chars_estimate: estimasi max karakter (safety limit)

    Returns:
        string teks watermark
    """
    img = Image.open(image_path).convert("RGB")
    flat = np.array(img, dtype=np.uint8).flatten()

    # Ambil LSB dari setiap nilai
    max_bits = num_chars_estimate * 8 + 8   # +8 untuk delimiter
    bits = [(int(val) & 1) for val in flat[:max_bits]]

    return bits_to_text(bits)


# ─────────────────────────────────────────────
# BAGIAN 3: JPEG COMPRESSION & EVALUASI
# ─────────────────────────────────────────────

def compress_jpeg(input_path: str, output_path: str, quality: int) -> int:
    """
    Kompres gambar menggunakan JPEG dengan quality factor tertentu.
    Pillow menggunakan skala QF 1–95 (bukan 100, karena 100 masih lossy).

    Args:
        input_path : gambar sumber
        output_path: path output .jpg
        quality    : quality factor (1=terburuk, 95=terbaik)

    Returns:
        ukuran file dalam bytes
    """
    img = Image.open(input_path).convert("RGB")
    img.save(output_path, format="JPEG", quality=quality, optimize=True)
    size = os.path.getsize(output_path)
    return size


def evaluate_lsb_robustness(watermarked_png: str, original_wm_text: str,
                              qf_list: list[int], output_dir: str) -> dict:
    """
    Evaluasi ketahanan watermark LSB terhadap kompresi JPEG.

    Untuk setiap QF:
    1. Kompres gambar ber-watermark ke JPEG
    2. Ekstrak watermark dari JPEG
    3. Bandingkan dengan watermark asli → hitung Bit Accuracy

    Bit Accuracy = jumlah bit benar / total bit × 100%

    Args:
        watermarked_png : path gambar PNG ber-watermark
        original_wm_text: teks watermark asli
        qf_list         : list quality factor yang akan diuji
        output_dir      : folder penyimpanan hasil kompresi

    Returns:
        dict hasil evaluasi per QF
    """
    os.makedirs(output_dir, exist_ok=True)
    original_bits = text_to_bits(original_wm_text)
    total_bits = len(original_bits)

    results = {}

    print(f"\n{'QF':>4} | {'File Size':>12} | {'Bit Accuracy':>13} | {'Extracted Text'}")
    print("-" * 70)

    for qf in qf_list:
        compressed_path = os.path.join(output_dir, f"lsb_compressed_qf{qf:02d}.jpg")
        file_size = compress_jpeg(watermarked_png, compressed_path, qf)

        extracted = extract_lsb(compressed_path, num_chars_estimate=len(original_wm_text) + 10)

        # Hitung bit accuracy
        extracted_bits = text_to_bits(extracted) if extracted else []
        min_len = min(len(original_bits), len(extracted_bits))

        if min_len == 0:
            accuracy = 0.0
        else:
            correct = sum(1 for a, b in zip(original_bits[:min_len], extracted_bits[:min_len]) if a == b)
            accuracy = (correct / total_bits) * 100

        results[qf] = {
            "file_size_bytes": file_size,
            "bit_accuracy": round(accuracy, 2),
            "extracted_text": extracted,
            "match": extracted == original_wm_text
        }

        status = "MATCH" if extracted == original_wm_text else "FAIL "
        print(f"{qf:>4} | {file_size:>10} B | {accuracy:>11.2f}% | {status} → '{extracted[:30]}'")

    return results


# ─────────────────────────────────────────────
# BAGIAN 4: PSNR — Ukur kualitas visual
# ─────────────────────────────────────────────

def calculate_psnr(original_path: str, watermarked_path: str) -> float:
    """
    Hitung PSNR (Peak Signal-to-Noise Ratio) antara gambar asli dan ber-watermark.
    PSNR tinggi = perbedaan kecil = watermark tidak terlihat.
    Biasanya PSNR > 40 dB dianggap sangat baik (tidak terlihat mata).

    Formula:
        MSE  = mean((original - watermarked)^2)
        PSNR = 10 * log10(255^2 / MSE)
    """
    orig = np.array(Image.open(original_path).convert("RGB"), dtype=np.float64)
    wm   = np.array(Image.open(watermarked_path).convert("RGB"), dtype=np.float64)

    mse = np.mean((orig - wm) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((255.0 ** 2) / mse)
    return round(psnr, 4)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    BASE     = os.path.dirname(os.path.abspath(__file__))
    IMG_IN   = os.path.join(BASE, "..", "face.jpg")
    WM_PNG   = os.path.join(BASE, "lsb_watermarked.png")
    OUT_DIR  = os.path.join(BASE, "results")

    WATERMARK_TEXT = "Copyright2024-MyName"

    print("=" * 70)
    print("LSB WATERMARKING — IMPLEMENTASI DARI NOL")
    print("=" * 70)

    # Step 1: Embed
    print("\n[STEP 1] Menyisipkan watermark...")
    embed_lsb(IMG_IN, WATERMARK_TEXT, WM_PNG)

    # Step 2: Verifikasi ekstraksi dari PNG (lossless)
    print("\n[STEP 2] Verifikasi ekstraksi dari PNG (lossless)...")
    extracted_png = extract_lsb(WM_PNG, num_chars_estimate=100)
    print(f"  Watermark asli : '{WATERMARK_TEXT}'")
    print(f"  Hasil ekstraksi: '{extracted_png}'")
    print(f"  Status: {'✓ BERHASIL' if extracted_png == WATERMARK_TEXT else '✗ GAGAL'}")

    # Step 3: PSNR
    print("\n[STEP 3] Menghitung PSNR...")
    psnr_val = calculate_psnr(IMG_IN, WM_PNG)
    print(f"  PSNR = {psnr_val} dB  (>40 dB = tidak terlihat mata)")

    # Step 4: Evaluasi robustness vs JPEG compression
    print("\n[STEP 4] Evaluasi robustness vs JPEG Quality Factor...")
    qf_list = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    results = evaluate_lsb_robustness(WM_PNG, WATERMARK_TEXT, qf_list, OUT_DIR)

    # Step 5: Ringkasan
    print("\n[STEP 5] RINGKASAN — QF minimum agar watermark masih bisa diekstrak:")
    for qf, r in sorted(results.items(), reverse=True):
        if r["match"]:
            print(f"  QF {qf}: watermark masih dapat diekstrak sempurna")
        else:
            print(f"  QF {qf}: watermark TIDAK dapat diekstrak (bit accuracy: {r['bit_accuracy']}%)")
