# Laporan Proyek Machine Learning - Ghifari Adil Ruchiyat

## 1. Domain Proyek

### Latar Belakang

Deteksi hunian (*occupation detection*) menentukan apakah suatu ruangan sedang dihuni oleh seseorang menggunakan data yang didapat dari sensor, misalnya sensor cahaya, suhu, kelembapan, atau CO2. Deteksi hunian merupakan komponen penting di bidang Internet of Things (IoT), terutama dalam konteks *smart building*. Deteksi hunian dapat digunakan untuk mengontrol bangunan secara lebih cerdas, misalnya untuk ventilasi, lampu, pemanas, dan pendingin pada bangunan. Deteksi hunian yang akurat pada sebuah bangunan dapat membantu dalam penghematan energi. Oleh karenanya, penting untuk menemukan cara mendeteksi hunian dari sebuah bangunan dengan efektif.

Teknik pembelajaran mesin atau *machine learning* telah banyak diterapkan dalam banyak bidang, yang mengarah pada peningkatan aplikasinya untuk analisis kinerja, prediksi, dan evaluasi sistem. Dalam konteks deteksi hunian, *machine learning* dapat digunakan untuk menganalisis data sensor yang dikumpulkan dan menentukan apakah suatu ruangan sedang dihuni atau tidak.

### Referensi Terkait

Sebuah studi berjudul ["Occupancy Detection in Room Using Sensor Data"](https://arxiv.org/abs/2101.03616) oleh Mohammadhossein Toutiaee memberikan solusi untuk deteksi hunian menggunakan data sensor. Studi tersebut menunjukkan bahwa pendeteksian hunian dalam suatu ruangan dapat dilakukan dengan menganalisis data yang dikumpulkan dan menggunakan *machine learning*.

Studi lain yang berjudul ["Promoting Occupancy Detection Accuracy Using On-Device Lifelong Learning"](https://ieeexplore.ieee.org/abstract/document/10081223) oleh Muhammad Emad-Ud-Din dan Ya Wang menyebutkan bahwa akurasi deteksi *machine learning* bergantung pada keragaman dataset yang dikumpulkan.

## 2. Business Understanding

### Problem Statements
Berdasarkan latar belakang di atas, perlu dikembangkan sebuah sistem deteksi hunian untuk menjawab permasalahan berikut:
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap deteksi hunian?
- Berdasarkan fitur tertentu, apa hasil deteksi hunian yang tepat?

### Goals
Untuk menjawab pertanyaan tersebut, *predictive modelling* dibuat dengan tujuan atau *goals* sebagai berikut:
- Menentukan fitur yang paling berpengaruh terhadap deteksi hunian
- Membuat model *machine learning* yang dapat memprediksi hunian seakurat mungkin berdasarkan fitur-fitur yang ada

### Solution statements
Untuk mencapai tujuan tersebut, hal yang perlu dilakukan adalah sebagai berikut:
- Membuat baseline model *machine learning*
- Membuat model dengan algoritma Random Forest dan melakukan hyperparameter tuning

## 3. Data Understanding

Data yang digunakan adalah data [Occupancy Detection](https://archive.ics.uci.edu/dataset/357/occupancy+detection) oleh Luis Candanedo. Data ini memiliki 20560 sampel data dengan 7 fitur, yakni date, Temperature, Humidity, Light, CO2, HumidityRatio, dan Occupancy

### Deskripsi Variable

Variable-variable yang ada pada dataset tersebut adalah sebagai berikut:

- date: merupakan waktu data diperoleh
- Temperature: merupakan suhu ruangan dalam satuan Celsius
- Humidity: merupakan kelembapan relatif dalam satuan persentase
- Light: merupakan intensitas penerangan cahaya dalam satuan Lux
- CO2: merupakan konsentrasi molekul karbon dioksida di udara dalam satuan ppm
- HumidityRatio: merupakan fitur turunan dari Temperature dan Humidity dalam satuan kg-uap-air/kg-udara
- Occupancy: merupakan deteksi hunian ruangan, didapat dari gambar yang diambil setiap menitnya. 0 artinya tidak berpenghuni, 1 artinya berpenghuni

Berikut adalah hasil `df.describe()` dari data:

![](pic/03-01.png)

### Exploratory Data Analysis

Berikut adalah hasil analisis dari data tersebut:

1. Plot kemunculan kelas

    Berikut adalah plot kemunculan kelas 0 (tidak berpenghuni) dan 1 (berpenghuni) pada fitur Occupancy

    ![](pic/03-02.png)

    Terlihat bahwa kelas 1 lebih sedikit muncul daripada kelas 0. Hal ini dapat mempengaruhi model.

2. Plot histogram fitur

    Berikut adalah grafik histogram pada masing-masing fitur numerik pada data

    ![](pic/03-03.png)

    Dari grafik diatas, terlihat bahwa fitur Light dan CO2 sebagian besar berukuran kecil, fitur Light banyak di bawah 100, dan CO2 banyak di bawah 600.

3. Plot histogram fitur tiap kelas

    Berikut adalah grafik histogram tiap fitur numerik pada data, dikelompokkan berdasarkan kelas

    ![](pic/03-04.png)

4. Boxplot

    Berikut adalah diagram boxplot dari masing-masing fitur numerik, dikelompokkan berdasarkan kelas

    ![](pic/03-05.png)

5. Correlation Matrix

    Metode Pearson, yang merupakan default pada `df.corr()`, umumnya digunakan untuk memeriksa korelasi dari dua fitur kontinyu. Pada data yang digunakan, fitur occupancy merupakan fitur diskret, sedangkan fitur lainnya adalah fitur kontinyu, sehingga kurang cocok menggunakan metode Pearson.

    Disini, metode tes korelasi yang digunakan adalah metode *Cramer's V* atau koefisien Cramer dengan rumus

    ![](pic/03-06.png)

    - χ2 = Nilai statistik Chi-square
    - n = Ukuran contoh total
    - r = Jumlah baris tabel kontingensi
    - c = Jumlah kolom tabel kontingensi

    Dengan menggunakan metode tersebut, didapat Correlation Matrix sebagai berikut:

    ![](pic/03-07.png)

    Dari matrix diatas, didapat bahwa fitur Occupancy berkorelasi tinggi dengan fitur Light, tertinggi kedua dengan fitur CO2, lalu HumidityRatio, kemudian Temperature, terakhir dengan Humidity. Semuanya berkorelasi cukup tinggi dengan nilai koefisiennya diatas 0.6

## 4. Data Preparation

1. Drop fitur HumidityRatio dan date

    Fitur date adalah fitur timestamp unik dari masing-masing data. Fitur HumidityRatio adalah fitur turunan dari humidity dan temperature. Kedua fitur didrop untuk mengurangi kompleksitas data dan model

    ```py
    dfset = df.drop(columns=['date', 'HumidityRatio'])
    ```

2. Membuang outlier

    Outlier pada data dari masing-masing class dibuang.

    ```py
    def is_outlier(group):
        Q1 = group.quantile(0.25)
        Q3 = group.quantile(0.75)
        IQR = Q3 - Q1

        lower_limit = Q1-1.5*IQR
        upper_limit = Q3+1.5*IQR
        
        return ~group.between(lower_limit, upper_limit)

    for column in ['Temperature', 'Humidity', 'Light', 'CO2']:
        dfset = dfset[~dfset.groupby('Occupancy', group_keys=False)[column].apply(is_outlier)]
    ```

    Kemudian kita periksa jumlah data yang tersisa dengan menggunakan kode berikut

    ```py
    dfset.shape
    ```

    Didapat hasilnya adalah sebagai berikut:

    ```
    (15838, 5)
    ```

3. Membagi data menjadi data training dan data validasi

    Data dibagi menjadi dua, yakni data training dan data validasi. Data training akan digunakan untuk pelatihan algoritma, sedangkan data validasi akan digunakan untuk evaluasi model. Data validasi sebesar 10% dari total data.

    ```py
    x = dfset.drop(columns=('Occupancy'))
    y = dfset.Occupancy
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 1)

    print(f'Jumlah data training: {len(xTrain)}')
    print(f'Jumlah data validasi: {len(xTest)}')
    ```

    Hasilnya adalah sebagai berikut:

    ```
    Jumlah data training: 14254
    Jumlah data validasi: 1584
    ```

4. Normalisasi Data

    ```py
    scaler = StandardScaler().fit(xTrain)

    xTrainScaled = scaler.transform(xTrain)
    xTestScaled = scaler.transform(xTest)
    ```

5. Oversampling dengan metode SMOTE

    Data kelas occupied (kelas 1) jauh lebih sedikit dari data kelas not occupied (kelas 0). Hal ini dapat memberikan bias pada model. Untuk mencegahnya, dilakukan oversampling dengan menggunakan meotde SMOTE

    ```py
    smote = SMOTE(random_state=1)
    xTrainResampled, yTrainResampled = smote.fit_resample(xTrain, yTrain) 
    ```

## 5. Modeling

Model yang digunakan untuk deteksi hunian ini adalah model Random Forest Classifier. Random Forest adalah model ensemble yang terdiri dari beberapa model decision tree. Tiap model decision tree memiliki hyperparameter yang berbeda dan dilatih pada beberapa bagian (subset) data yang berbeda. Prediksi akhir diambil dari prediksi terbanyak pada seluruh tree.

Kelebihan dari algoritma ini adalah:
- Mampu menangani noise dan variasi dalam data
- Dapat menangani data non-linear dengan baik
- Risiko overfitting lebih rendah
- Akurasi yang lebih baik daripada algoritma klasifikasi lainnya

Sedangkan kekurangan dari algoritma ini adalah:
- Sulit menginterpretasikan pengaruh setiap fitur dalam membuat keputusan
- Sulit menangani data dengan fitur yang sangat banyak atau data berdimensi tinggi
- Membutuhkan banyak proses komputasi
- Waktu komputasi pada dataset berskala besar relatif lambat

Proses improvement pada model akan dilakukan dengan hyperparameter tuning. Tuning ini akan dilakukan dengan menggunakan random search. Hyperparameter yang akan di-tune adalah sebagai berikut:

- n_estimator (jumlah tree pada model): 100, 150, 200, 250, dst sampai 500
- max_depth (maksimal depth tiap tree): 3 sampai 20

Tahapan melakukan modellingnya adalah sebagai berikut:

1. Membuat baseline model

    Parameter *random_state* digunakan agar dapat memberikan hasil konsisten setiap kali menjalankan notebook

    ```py
    baselineModel = RandomForestClassifier(random_state=1)
    ```

2. Melatih baseline model

    Model dilatih dengan training data yang sudah dilakukan SMOTE

    ```py
    baselineModel.fit(xTrainResampled, yTrainResampled)
    ```

3. Inisiasi Random Search

    ```py
    paramGrid = {
        'n_estimators': range(100, 500, 50),
        'max_depth': range(3, 20),
    }

    search = RandomizedSearchCV(RandomForestClassifier(random_state=1), paramGrid, random_state=1)
    ```

4. Melakukan Random Search pada model Random Forest

    ```py
    search.fit(xTrainResampled, yTrainResampled)
    ```

    Hasil parameter terbaik pada random search tersebut adalah sebagai berikut:

    ```py
    search.best_params_
    ```

    Hasilnya adalah sebagai berikut:

    ```
    {'n_estimators': 200, 'max_depth': 8}
    ```

## Evaluation

Metrik evaluasi model yang akan digunakan adalah sebagai berikut:

1. Accuracy

    Accuracy adalah persentase hasil prediksi benar oleh model. 

Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**