# Arsitektur dan Sistem Komputer
Analisis Performa Median Filter pada Pengolahan Citra Digital Menggunakan Sequential Computing, OpenMP, dan OpenCL

## Anggota Kelompok

1. Bibika Firmansyah - 25032014083
2. Dika Ahmad Saputra - 25032014070
3. Saifur Rofi'i - 25032014024

## Deskripsi Proyek

Proyek ini mengimplementasikan Median Filter menggunakan Sequential Computing, OpenMP, dan OpenCL untuk mengurangi noise pada citra digital serta membandingkan performa ketiga metode berdasarkan waktu eksekusi dan speedup.

## Fitur Utama

- Membaca citra input
- Menambahkan Salt and Pepper Noise
- Menghilangkan noise menggunakan Median Filter
- Implementasi Sequential
- Implementasi OpenMP
- Implementasi OpenCL
- Pengukuran waktu eksekusi
- Perbandingan performa

## Langkah-langkah Menjalankan

### Prasyarat

| Tool | Keterangan |
|------|-----------|
| GCC ≥ 7 | Compiler C (disertakan di sebagian besar distro Linux) |
| OpenMP | Sudah bawaan GCC (flag `-fopenmp`) |
| OpenCL | Driver GPU (NVIDIA/AMD/Intel) + header `libOpenCL-dev` |
| Python 3 | Untuk generate gambar uji & benchmark script |

**Install dependensi (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install gcc make python3 ocl-icd-opencl-dev opencl-headers
```

**Untuk NVIDIA GPU, pastikan driver CUDA terinstall:**
```bash
nvidia-smi   # verifikasi driver aktif
```

---

## Struktur File

```
src/
├── sequential.c          ← Implementasi Sequential
├── openmp_filter.c       ← Implementasi OpenMP (CPU Parallel)
├── opencl_filter.c       ← Implementasi OpenCL (GPU Parallel)
└── generate_test_image.py← Script buat gambar PGM uji

```

---

## LANGKAH 0 — Siapkan Gambar Input

Buat gambar PGM grayscale 512×512 sebagai input:

```bash
python3 generate_test_image.py 512 512 input.pgm
```

Atau gunakan gambar PGM Anda sendiri (harus format P5 binary).
Untuk konversi dari JPG/PNG:
```bash
# Jika punya ImageMagick:
convert foto.jpg -type Grayscale input.pgm
```

---

## LANGKAH 1 — Sequential (Baseline)

### 1a. Compile
```bash
gcc -O2 -o sequential sequential.c -lm
```

### 1b. Jalankan
```bash
./sequential input.pgm output_seq.pgm 0.05 3
```

**Parameter:**
| Parameter | Contoh | Keterangan |
|-----------|--------|-----------|
| `input.pgm` | `input.pgm` | Gambar input PGM |
| `output.pgm` | `output_seq.pgm` | Gambar hasil output |
| `noise_ratio` | `0.05` | Intensitas noise (0.0–1.0, mis. 0.05 = 5%) |
| `kernel_size` | `3` | Ukuran kernel filter (3, 5, 7 — harus ganjil) |

### 1c. Output yang diharapkan
```
=== Sequential Median Filter ===
[Step 1] Image loaded: 512x512 pixels
[Step 2] Salt-and-pepper noise added (ratio=5.00%)
[Step 3] Noisy image saved: noisy_output_seq.pgm


  RESULTS – Sequential
------------------------------
  Image size   : 512 x 512
  Kernel size  : 3 x 3
  Noise ratio  : 5.0%
  Exec time    : 142.3500 ms    ← waktu eksekusi
  PSNR (noisy) : 20.45 dB
  PSNR (filter): 30.12 dB
  Output saved : output_seq.pgm
------------------------------
```

**Catat nilai "Exec time" untuk perbandingan!**

---

## LANGKAH 2 — OpenMP (CPU Parallel)

### 2a. Compile
```bash
gcc -O2 -fopenmp -o openmp_filter openmp_filter.c -lm
```

### 2b. Jalankan (dengan 4 thread)
```bash
./openmp_filter input.pgm output_omp.pgm 0.05 3 4
```

**Parameter tambahan:**
| Parameter | Contoh | Keterangan |
|-----------|--------|-----------|
| `num_threads` | `4` | Jumlah thread (opsional, default = max CPU) |

### 2c. Coba variasi thread
```bash
# 1 thread (baseline OpenMP overhead)
./openmp_filter input.pgm output_omp.pgm 0.05 3 1

# 2 thread
./openmp_filter input.pgm output_omp.pgm 0.05 3 2

# 4 thread
./openmp_filter input.pgm output_omp.pgm 0.05 3 4

# 8 thread (jika CPU mendukung)
./openmp_filter input.pgm output_omp.pgm 0.05 3 8
```

### 2d. Output yang diharapkan
```
=== OpenMP Median Filter ===
Threads      : 4 (max available: 8)

[Step 4] Benchmarking across thread counts...
Threads    Exec Time (ms)   PSNR (noisy) PSNR (filt)
-------    --------------   ------------ ----------
1          138.2400         20.45        30.12
2          72.1100          20.45        30.12
4          38.5600          20.45        30.12
8          22.3400          20.45        30.12


  RESULTS – OpenMP
------------------------------
  Best threads   : 8
  Best exec time : 22.3400 ms
  Speedup vs 1T  : ~6.2x
------------------------------
```

---

## LANGKAH 3 — OpenCL (GPU Parallel)

### 3a. Compile (Windows)
```bash
gcc -I"Path/to/vcpkg/installed/x64-windows/include" opencl_filter -L"Path/to/vcpkg/installed/x64-windows/lib" -lOpenCL -fopenmp -DCL_TARGET_OPENCL_VERSION=120 -Wno-deprecated-declarations -o opencl_filter
```

### 3c. Jalankan
```bash
./opencl_filter input.pgm output_ocl.pgm 0.05 3
```

> **Catatan:** OpenCL kernel_size maksimum 11×11. Jika tidak ada GPU, program otomatis fallback ke CPU OpenCL device.

### 3d. Output yang diharapkan
```
=== OpenCL Median Filter ===
[OpenCL] Using device: NVIDIA GeForce RTX 3060


  RESULTS – OpenCL
------------------------------
  GPU kernel time   : 2.4100 ms   ← waktu kernel GPU saja
  Total time (H2D+kernel+D2H) : 18.5300 ms
  PSNR (noisy)      : 20.45 dB
  PSNR (filter)     : 30.12 dB
  Output saved      : output_ocl.pgm
------------------------------   
```

---

## Tips Eksperimen Lanjutan

### Coba ukuran gambar berbeda
```bash
python3 generate_test_image.py 256 256 small.pgm
python3 generate_test_image.py 1024 1024 large.pgm
python3 generate_test_image.py 2048 2048 xlarge.pgm
```

### Coba kernel size berbeda
```bash
# Kernel 5x5 (lebih halus, lebih lambat)
./sequential  input.pgm out_seq_k5.pgm 0.05 5
./openmp_filter input.pgm out_omp_k5.pgm 0.05 5 4
./opencl_filter input.pgm out_ocl_k5.pgm 0.05 5

# Kernel 7x7
./sequential  input.pgm out_seq_k7.pgm 0.05 7
```

### Coba noise ratio berbeda
```bash
./sequential input.pgm out_seq_10.pgm 0.10 3   # 10% noise
./sequential input.pgm out_seq_20.pgm 0.20 3   # 20% noise
./sequential input.pgm out_seq_50.pgm 0.50 3   # 50% noise (berat)
```

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `libOpenCL.so not found` | `sudo apt install ocl-icd-opencl-dev` |
| `CL_DEVICE_NOT_FOUND` | Pastikan driver GPU terinstall; program akan fallback ke CPU |
| `Only binary PGM (P5) supported` | Konversi gambar dengan: `convert image.png -type Grayscale out.pgm` |
| Hasil PSNR sama semua | Normal — semua metode menghasilkan output identik secara matematis |
| OpenMP lebih lambat dari Sequential | Overhead thread > keuntungan paralel di gambar kecil; coba gambar 1024×1024 |

---

## Penjelasan Metrik

| Metrik | Keterangan |
|--------|-----------|
| **Exec time (ms)** | Waktu murni filter berjalan |
| **Speedup** | Exec time Sequential / Exec time metode ini |
| **PSNR (dB)** | Peak Signal-to-Noise Ratio — makin tinggi makin baik. >30 dB = kualitas baik |
| **H2D / D2H** | Host-to-Device / Device-to-Host (transfer memori CPU↔GPU pada OpenCL) |


## Link Vidio Penjelasan
