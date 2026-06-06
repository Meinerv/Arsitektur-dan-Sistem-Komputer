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

### 1. Prasyarat

   - Kompiler C/C++ yang mendukung OpenMP (misalnya GCC atau G++).
   - OpenCL SDK telah terinstal pada sistem.
   - Driver GPU yang mendukung OpenCL telah terpasang dan berfungsi dengan baik.
   - File kernel OpenCL tersedia pada direktori yang ditentukan (`src/kernel.cl`).
   - Sistem operasi Windows atau Linux.    

### 2. Kompilasi
   
   Sequential Version
   - Masuk ke direktori `src/sequential`.
   - kompilasi program: `gcc median_sequential.c -o median_seq`.
      
   OpenMP Version
   - Masuk ke direktori `src/openmp`.
   - Kompilasi program: `gcc median_openmp.c -fopenmp -o median_omp`.
      
   OpenCL Version
   - Masuk ke direktori `src/opencl`
   - Kompilasi program: `gcc median_opencl.c -lOpenCL -o median_cl`.

### 3. Ekseskusi

   - Sequential: `median_seq.exe input.jpg output_seq.jpg`
   - OpenMP: `median_omp.exe input.jpg output_omp.jpg`
   - OpenCL: `median_cl.exe input.jpg output_cl.jpg`

## Hasil Pengujian


## Link Vidio Penjelasan
