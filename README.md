# 🔐 Watermarking Invisible + JPEG Compression

## 📁 Struktur Proyek

```
Watermarking-Invisible/
├── Face.jpg                    # Gambar input (foto wajah)
├── LSB/
│   ├── LSB_Watermark.py        # Implementasi LSB lengkap
├── DCT/
│   ├── DCT_Watermark.py        # Implementasi DCT lengkap
└── README.md
```

---

## ⚙️ Requirements

```bash
pip install numpy Pillow
```

---

## Metode 1 — LSB (Least Significant Bit)

### Konsep Dasar

Setiap piksel RGB memiliki 3 channel (R, G, B), masing-masing bernilai 0–255 (8 bit).  
Bit paling kecil (**LSB**) hanya berkontribusi nilai **±1** terhadap warna — perubahan ini **tidak terlihat mata manusia**.

LSB watermarking menyembunyikan pesan dengan **mengganti bit terakhir** setiap nilai channel piksel dengan bit watermark.

```
Piksel asli  :  11001010  (202)
Watermark bit:           1
Piksel baru  :  11001011  (203)  ← perubahan hanya 1 nilai
```

### Pipeline Lengkap

```
Gambar Asli (RGB)
      │
      ▼
Konversi Teks → Bit (ASCII 8-bit per karakter)
      │  "Hello" → 01001000 01100101 01101100 01101100 01101111
      │  + Delimiter (11111110) sebagai penanda akhir pesan
      ▼
Flatten Piksel (H × W × 3 → array 1D)
      │
      ▼
Untuk setiap bit watermark [idx]:
    pixel[idx] = (pixel[idx] AND 11111110) OR bit_watermark
      │          ↑ clear LSB                ↑ set dengan bit baru
      ▼
Reshape kembali → (H, W, 3)
      │
      ▼
Simpan sebagai PNG (lossless — wajib, JPEG akan merusak LSB)
```

### Cara Kerja Encoding (Detail)

**Step 1: Konversi teks ke bit**
```python
# Contoh: "Hi" → bits
'H' = ASCII 72  = 01001000
'i' = ASCII 105 = 01101001
Delimiter = 0xFE = 11111110

bits = [0,1,0,0,1,0,0,0, 0,1,1,0,1,0,0,1, 1,1,1,1,1,1,1,0]
```

**Step 2: Modifikasi LSB piksel**
```python
# Untuk setiap bit watermark pada indeks idx:
pixel[idx] = (pixel[idx] & 0b11111110) | watermark_bit
#             ↑ hapus LSB lama          ↑ sisipkan bit baru
```

**Step 3: PSNR sangat tinggi**  
Karena maksimal error per piksel = 1, PSNR bisa mencapai **94+ dB** (hampir tidak ada perbedaan visual).

### Cara Kerja Decoding

```python
# Baca LSB dari setiap nilai piksel secara berurutan
bit = pixel[idx] & 1

# Kumpulkan 8 bit → 1 karakter ASCII
# Berhenti saat delimiter 0xFE ditemukan
```

### Menjalankan LSB

```bash
python lsb/lsb_watermark.py
```

**Output yang dihasilkan:**
```
[STEP 1] Menyisipkan watermark...
[LSB] Watermark berhasil disimpan: lsb_watermarked.png
[LSB] Bit disisipkan: 168 bit (21 karakter + delimiter)

[STEP 2] Verifikasi dari PNG...
  Watermark asli : 'Copyright2024-MyName'
  Hasil ekstraksi: 'Copyright2024-MyName'
  Status: ✓ BERHASIL

[STEP 3] PSNR = 94.712 dB  (>40 dB = tidak terlihat mata)
```

### Hasil Evaluasi Robustness LSB vs JPEG

| QF  | File Size  | Bit Accuracy | Status  |
|-----|-----------|--------------|----------|
| 95  | 112,103 B | 50.60%       | GAGAL    |
| 90  | 91,938 B  | 47.02%       | GAGAL    |
| 80  | 80,014 B  | 48.21%       | GAGAL    |
| 70  | 69,471 B  | 58.33%       | GAGAL    |
| 60  | 64,435 B  | 6.55%        | GAGAL    |
| 50  | 60,392 B  | 48.21%       | GAGAL    |
| 40  | 52,322 B  | 32.14%       | GAGAL    |
| 30  | 35,696 B  | 52.98%       | GAGAL    |
| 20  | 30,290 B  | 42.86%       | GAGAL    |
| 10  | 19,713 B  | 0.00%        | GAGAL    |

> **Kesimpulan LSB**: Watermark **tidak dapat diekstrak** setelah kompresi JPEG.  
> Ini bukan bug — ini sifat fundamental LSB. JPEG adalah *lossy compression* yang mengubah nilai piksel secara agresif, sehingga LSB (yang hanya menyimpan 1 bit) langsung hilang.  
> LSB hanya cocok untuk format **lossless** (PNG, BMP, TIFF).

---

## Metode 2 — DCT (Discrete Cosine Transform)

### Konsep Dasar

DCT mentransformasi gambar dari **domain spasial (piksel)** ke **domain frekuensi (koefisien)**.  
Ide kuncinya: **watermark disisipkan di koefisien frekuensi menengah** — bagian yang kurang sensitif terhadap kompresi JPEG dibanding frekuensi tinggi, namun tidak sepenting frekuensi rendah (yang menjaga kualitas visual utama).

```
Domain Spasial          Domain Frekuensi (DCT)
┌────────────┐           ┌────────────┐
│  piksel    │  → DCT →  │ DC │ rendah│
│  256 180   │           ├────┼───────┤
│  200 240   │           │ menengah  │ ← watermark di sini
│    ...     │           ├───────────┤
└────────────┘           │  tinggi   │
                         └────────────┘
```

### Matematika DCT

**Formula DCT-2D:**

$$F(u,v) = c(u) \cdot c(v) \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} f(x,y) \cos\!\left[\frac{\pi u(2x+1)}{2N}\right] \cos\!\left[\frac{\pi v(2y+1)}{2N}\right]$$

di mana $c(0) = \frac{1}{\sqrt{N}}$, $c(k) = \sqrt{\frac{2}{N}}$ untuk $k > 0$.

**Implementasi menggunakan matriks transformasi:**

Daripada menghitung loop dua kali, kita bangun matriks $D$ berukuran $N \times N$:
```
D[0, x] = 1/√N
D[k, x] = √(2/N) * cos(π*k*(2x+1)/(2N))   untuk k > 0
```

Kemudian:
```
DCT(block)  = D  @ block @ D.T
IDCT(block) = D.T @ block @ D
```

Matriks $D$ dibangun sekali, lalu dipakai untuk semua blok → jauh lebih efisien.

### Pipeline Lengkap

```
Gambar Asli (RGB)
      │
      ▼
Konversi ke YCbCr → ambil channel Y (luminance/kecerahan)
      │  Alasan: mata lebih sensitif ke kecerahan, embed di Y
      │  membuat watermark lebih tidak terlihat
      ▼
Bagi Y menjadi blok 8×8
      │  Gambar 960×1280 → 120×160 = 19.200 blok
      ▼
DCT setiap blok: F = D @ block @ D.T
      │
      ▼
Generate watermark sequence (PN sequence bipolar +1/-1)
      │  dari teks watermark + seed acak
      ▼
Modifikasi koefisien frekuensi menengah (zigzag index 10–17):
      │  F'[u,v] = F[u,v] + alpha × w_i
      │  alpha = kekuatan watermark (default: 10)
      ▼
IDCT setiap blok: f' = D.T @ F' @ D
      │
      ▼
Gabungkan blok → channel Y baru → rekonstruksi RGB
      │
      ▼
Simpan sebagai PNG
```

### Urutan Zigzag dan Pemilihan Koefisien

Standar JPEG menggunakan urutan zigzag untuk mengunjungi koefisien DCT dari frekuensi rendah ke tinggi:

```
Urutan Zigzag 8×8:

(0,0)→(0,1)→(1,0)→(2,0)→(1,1)→(0,2)→...
  DC   ← frekuensi rendah →    ← menengah →    ← tinggi →

Indeks 0    : DC coefficient (jangan diubah — menentukan kecerahan rata-rata)
Indeks 1–9  : Frekuensi rendah (penting untuk kualitas visual)
Indeks 10–17: Frekuensi menengah ← WATERMARK DISISIPKAN DI SINI
Indeks 18+  : Frekuensi tinggi (hilang pertama saat JPEG)
```

### Embedding Formula

```
F'(u,v) = F(u,v) + α × wᵢ

di mana:
  F(u,v) = koefisien DCT asli di posisi (u,v)
  α      = alpha/strength (default: 10.0)
  wᵢ     = elemen ke-i dari watermark sequence (+1 atau -1)
```

### Deteksi (Non-Blind Correlation)

Karena DCT menggunakan sinyal spread-spectrum, deteksi dilakukan dengan **korelasi**:

```
1. Hitung DCT blok dari gambar asli dan gambar ter-watermark
2. Selisih koefisien = sinyal watermark yang tersisa setelah kompresi
3. Korelasikan dengan expected sequence:

   score = dot(extracted, expected) / (|extracted| × |expected|)

4. Jika score > threshold (0.3) → watermark terdeteksi
```

### Menjalankan DCT

```bash
python dct/dct_watermark.py
```

**Output yang dihasilkan:**
```
[STEP 1] Menyisipkan watermark DCT...
[DCT] 960x1280, blok 120x160=19200, kapasitas=153600 bit, alpha=10.0

[STEP 2] Verifikasi dari PNG...
  Correlation: 0.9996  | Status: ✓ BERHASIL

[STEP 3] PSNR = 36.0008 dB
```

### Hasil Evaluasi Robustness DCT vs JPEG

| QF  | File Size   | Correlation | Status     |
|-----|------------|-------------|-------------|
| 95  | 220,684 B  | 0.9958      | DETECTED    |
| 90  | 167,274 B  | 0.9901      | DETECTED    |
| 80  | 136,602 B  | 0.9623      | DETECTED    |
| 70  | 114,461 B  | 0.9194      | DETECTED    |
| 60  | 105,823 B  | 0.8617      | DETECTED    |
| 50  | 84,638 B   | 0.7128      | DETECTED    |
| 40  | 60,447 B   | 0.4007      | DETECTED    |
| **30**  | **39,026 B**   | **0.2619**      | **NOT FOUND**   |
| 20  | 31,915 B   | 0.1955      | NOT FOUND   |
| 10  | 19,917 B   | 0.0668      | NOT FOUND   |

> **Kesimpulan DCT**: Watermark **terdeteksi hingga QF 40**.  
> Di bawah QF 40 (kompresi sangat agresif), koefisien frekuensi menengah juga ikut terdegradasi sehingga sinyal watermark terlalu lemah untuk dideteksi.

---

## 📊 Perbandingan LSB vs DCT

| Aspek | LSB | DCT |
|-------|-----|-----|
| **PSNR** | ~94 dB (sangat tinggi) | ~36 dB (masih baik) |
| **Invisibility** | ✓ Sangat tidak terlihat | ✓ Tidak terlihat |
| **Robust vs JPEG** | ✗ Langsung hilang | ✓ Bertahan hingga QF 40 |
| **Kompleksitas** | Sangat sederhana | Menengah |
| **Domain kerja** | Spasial (piksel) | Frekuensi (koefisien) |
| **Metode deteksi** | Blind (tanpa gambar asli) | Non-blind (butuh original) |
| **Kapasitas** | Tinggi (1 bit/channel) | Lebih rendah (terikat blok) |
| **Format output aman** | PNG/BMP/TIFF | PNG/BMP/TIFF |

### Mengapa LSB Gagal di JPEG?

JPEG bekerja dalam domain DCT. Saat kompresi, JPEG:
1. Bagi gambar ke blok 8×8
2. Transformasi DCT
3. **Quantization** — koefisien dibagi nilai tertentu dan dibulatkan (lossy!)
4. IDCT untuk kembali ke piksel

Proses quantization **mengubah nilai piksel secara signifikan** (bisa ±10 atau lebih).  
LSB hanya menyimpan 1 bit (perubahan nilai ±1) → langsung terhapus.

### Mengapa DCT Lebih Robust?

DCT watermarking menyisipkan perubahan langsung di **koefisien frekuensi**,  
di layer yang sama dengan cara JPEG menyimpan data.  
Selama alpha cukup besar dan koefisien yang dipilih tidak terlalu di-quantize habis,  
sinyal watermark masih bisa dideteksi via korelasi.

---

## 🔧 Parameter Tuning

### LSB
- `watermark_text`: teks yang disembunyikan
- Tidak ada parameter lain yang perlu diubah

### DCT
- `alpha`: kekuatan embedding
  - Nilai kecil (5–8): PSNR lebih tinggi, kurang robust
  - Nilai besar (10–20): lebih robust, PSNR sedikit turun
- `WM_COEF_INDICES`: koefisien zigzag yang dipakai (default: 10–17)
  - Geser ke indeks lebih kecil: lebih robust tapi lebih terlihat
  - Geser ke indeks lebih besar: kurang robust, lebih tidak terlihat

---

## 📐 Metrik Evaluasi

### PSNR (Peak Signal-to-Noise Ratio)
Mengukur seberapa mirip gambar ber-watermark dengan aslinya.

$$\text{PSNR} = 10 \log_{10}\!\left(\frac{255^2}{\text{MSE}}\right) \quad \text{dB}$$

| PSNR | Interpretasi |
|------|--------------|
| > 40 dB | Tidak terlihat mata manusia |
| 36–40 dB | Hampir tidak terlihat |
| 30–36 dB | Sedikit terlihat jika dibandingkan langsung |
| < 30 dB | Terlihat dengan mata |

### Bit Accuracy (LSB)
Persentase bit yang berhasil diekstrak dengan benar setelah kompresi.
```
Bit Accuracy = (bit benar / total bit watermark) × 100%
```
Nilai ~50% berarti random noise — watermark sudah tidak bisa dipulihkan.

### Correlation Score (DCT)
Korelasi Pearson ternormalisasi antara sinyal yang diekstrak dan yang diharapkan.
```
score = dot(extracted, expected) / (|extracted| × |expected|)
```
Rentang: -1 hingga +1. Threshold deteksi: 0.3 (empiris).

---

## 🏃 Jalankan Semua Sekaligus

```bash
# LSB
python lsb/lsb_watermark.py

# DCT
python dct/dct_watermark.py
```

---
