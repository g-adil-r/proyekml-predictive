# Laporan Proyek Machine Learning - Ghifari Adil Ruchiyat

## 1. Domain Proyek

### Latar Belakang

&nbsp;&nbsp;&nbsp;&nbsp;Deteksi hunian (*occupation detection*) menentukan apakah suatu ruangan sedang dihuni oleh seseorang menggunakan data yang didapat dari sensor, misalnya sensor cahaya, suhu, kelembapan, atau CO2. Deteksi hunian merupakan komponen penting di bidang Internet of Things (IoT), terutama dalam konteks *smart building*. Deteksi hunian dapat digunakan untuk mengontrol bangunan secara lebih cerdas, misalnya untuk ventilasi, lampu, pemanas, dan pendingin pada bangunan. Deteksi hunian yang akurat pada sebuah bangunan dapat membantu dalam penghematan energi. Oleh karenanya, penting untuk menemukan cara mendeteksi hunian dari sebuah bangunan dengan efektif.

&nbsp;&nbsp;&nbsp;&nbsp;Teknik pembelajaran mesin atau *machine learning* telah banyak diterapkan dalam banyak bidang, yang mengarah pada peningkatan aplikasinya untuk analisis kinerja, prediksi, dan evaluasi sistem. Dalam konteks deteksi hunian, *machine learning* dapat digunakan untuk menganalisis data sensor yang dikumpulkan dan menentukan apakah suatu ruangan sedang dihuni atau tidak.

### Referensi Terkait

&nbsp;&nbsp;&nbsp;&nbsp;Sebuah studi berjudul ["Occupancy Detection in Room Using Sensor Data"](https://arxiv.org/abs/2101.03616) oleh Mohammadhossein Toutiaee memberikan solusi untuk deteksi hunian menggunakan data sensor. Studi tersebut menunjukkan bahwa pendeteksian hunian dalam suatu ruangan dapat dilakukan dengan menganalisis data yang dikumpulkan dan menggunakan *machine learning*.

&nbsp;&nbsp;&nbsp;&nbsp;Studi lain yang berjudul ["Promoting Occupancy Detection Accuracy Using On-Device Lifelong Learning"](https://ieeexplore.ieee.org/abstract/document/10081223) oleh Muhammad Emad-Ud-Din dan Ya Wang menyebutkan bahwa akurasi deteksi *machine learning* bergantung pada keragaman dataset yang dikumpulkan.

## 2. Business Understanding

### Problem Statements
&nbsp;&nbsp;&nbsp;&nbsp;Berdasarkan latar belakang di atas, perlu dikembangkan sebuah sistem deteksi hunian untuk menjawab permasalahan berikut:
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap deteksi hunian?
- Berdasarkan fitur tertentu, apa hasil deteksi hunian yang tepat?

### Goals
&nbsp;&nbsp;&nbsp;&nbsp;Untuk menjawab pertanyaan tersebut, *predictive modelling* dibuat dengan tujuan atau *goals* sebagai berikut:
- Menentukan fitur yang paling berpengaruh terhadap deteksi hunian
- Membuat model *machine learning* yang dapat memprediksi hunian seakurat mungkin berdasarkan fitur-fitur yang ada

### Solution statements
&nbsp;&nbsp;&nbsp;&nbsp;Untuk mencapai tujuan tersebut, hal yang perlu dilakukan adalah sebagai berikut:
- Membuat baseline model *machine learning*
- Membuat model dengan algoritma Random Forest dan melakukan hyperparameter tuning

## 3. Data Understanding

&nbsp;&nbsp;&nbsp;&nbsp;Data yang digunakan adalah data [Occupancy Detection](https://archive.ics.uci.edu/dataset/357/occupancy+detection) oleh Luis Candanedo. 

### Deskripsi Variable

Variable-variable yang ada pada dataset tersebut adalah sebagai berikut:

- Temperature: merupakan suhu ruangan dalam satuan Celsius
- Humidity: merupakan kelembapan relatif dalam satuan persentase
- Light: merupakan intensitas penerangan cahaya dalam satuan Lux
- CO2: merupakan konsentrasi molekul karbon dioksida di udara dalam satuan ppm
- Humidity Ratio: merupakan fitur turunan dari Temperature dan Humidity dalam satuan kgwater-vapor/kg-air
- Occupancy: merupakan deteksi hunian ruangan, didapat dari gambar yang diambil setiap menitnya. 0 artinya tidak berpenghuni, 1 artinya berpenghuni

### Exploratory Data Analysis

Berikut adalah hasil analisis dari data tersebut:

1. Plot kemunculan kelas

    Berikut adalah plot kemunculan kelas 0 (tidak berpenghuni) dan 1 (berpenghuni) pada fitur Occupancy

    ![](pic/03-01.png)

    Terlihat bahwa kelas 1 lebih sedikit muncul daripada kelas 0. Hal ini dapat mempengaruhi model.

2. Plot histogram fitur

    Berikut adalah grafik histogram pada masing-masing fitur numerik pada data

    ![](pic/03-02.png)

    Dari grafik diatas, terlihat bahwa fitur Light dan CO2 sebagian besar berukuran kecil, fitur Light banyak di bawah 100, dan CO2 banyak di bawah 600.

3. Boxplot

    Berikut adalah diagram boxplot dari masing-masing fitur numerik, dikelompokkan berdasarkan kelas

    ![](pic/03-03.png)

4. Plot fitur-fitur dibandingkan dengan fitur occupancy

    Berikut adalah grafik plot masing-masing fitur numerik, dibandingkan dengan fitur occupancy. Garis biru menandakan data dari kelas occupied (kelas 1), sedangkan garis jingga menandakan data dari kelas not occupied (kelas 0)

    ![](pic/03-04.png)

    Disini, terdapat perbedaan data pada kelas occupied dan not occupied, terutama pada fitur light dan CO2.

5. Correlation Matrix

    Metode Pearson, yang merupakan default pada `df.corr()`, umumnya digunakan untuk memeriksa korelasi dari dua fitur kontinyu. Pada data yang digunakan, fitur occupancy merupakan fitur diskret, sedangkan fitur lainnya adalah fitur kontinyu, sehingga kurang cocok menggunakan metode Pearson.

    Disini, metode tes korelasi yang digunakan adalah metode *Cramer's V* atau koefisien Cramer dengan rumus

    ![](pic/03-05.png)

    - χ2 = Nilai statistik Chi-square
    - n = Ukuran contoh total
    - r = Jumlah baris tabel kontingensi
    - c = Jumlah kolom tabel kontingensi

    Dengan menggunakan metode tersebut, didapat Correlation Matrix sebagai berikut:

    ![](pic/03-06.png)

    Dari matrix diatas, didapat bahwa fitur Occupancy berkorelasi tinggi dengan fitur Light, tertinggi kedua dengan fitur CO2, lalu HumidityRatio, kemudian Temperature, terakhir dengan Humidity.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**