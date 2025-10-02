# Coba Craft dan OCR

Aplikasi konsol interaktif berbasis Python untuk mendeteksi teks pada gambar menggunakan metode segmentasi CRAFT. Setiap kata akan diberi bounding box dan disimpan sebagai gambar baru.

## Persiapan

1. Buat virtual environment (opsional tetapi disarankan).
2. Instal dependensi:

   ```bash
   pip install -r requirements.txt
   ```

## Menjalankan Aplikasi

Jalankan skrip utama kemudian ikuti instruksi pada terminal:

```bash
python main.py
```

Aplikasi akan meminta path gambar, mendeteksi teks menggunakan model CRAFT, dan membuat salinan gambar dengan bounding box berwarna merah untuk setiap kata yang terdeteksi.

## Catatan

- Proses deteksi membutuhkan koneksi internet pertama kali dijalankan untuk mengunduh bobot model CRAFT.
- Model dijalankan di CPU (tanpa CUDA) sehingga kecepatan bergantung pada spesifikasi perangkat Anda.
