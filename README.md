# Laporan Proyek Machine Learning - Nurma Intan Harianja

## Domain Proyek

Proyek ini berfokus pada **Prediksi Harga Taksi** menggunakan algoritma machine learning. Tujuan dari proyek ini adalah untuk memprediksi harga perjalanan taksi berdasarkan fitur-fitur yang ada seperti jarak perjalanan, waktu, kondisi cuaca, dan tarif per kilometer serta per menit. Data yang digunakan berasal dari dataset yang mengandung informasi mengenai perjalanan taksi yang mencakup variabel numerik dan kategorikal.

#### Mengapa dan Bagaimana Masalah Tersebut Harus Diselesaikan

Masalah prediksi harga taksi yang akurat sangat penting untuk meningkatkan efisiensi dalam sektor transportasi, terutama dalam hal penetapan harga yang adil bagi konsumen dan pengemudi. Penggunaan algoritma machine learning dalam memprediksi harga perjalanan dapat mengurangi ketidakpastian yang sering muncul pada sistem penetapan harga manual atau berbasis aturan tetap.

Dalam konteks ini, penyelesaian masalah tersebut bertujuan untuk menciptakan model yang mampu mengestimasi harga perjalanan taksi dengan mempertimbangkan berbagai faktor yang memengaruhi biaya perjalanan, seperti **jarak tempuh**, **durasi perjalanan**, **kondisi cuaca**, **waktu dalam sehari**, dan **kondisi lalu lintas**. Dengan menggunakan algoritma yang tepat, seperti **Random Forest**, dapat diperoleh prediksi harga yang lebih akurat dan efisien dibandingkan metode konvensional, serta membantu meningkatkan transparansi dalam sistem tarif taksi.

Untuk menyelesaikan masalah ini, dilakukan beberapa tahap penting, yaitu:

1.  **Pengolahan data** untuk memastikan bahwa data yang digunakan bebas dari nilai yang hilang dan anomali.
2.  **Penggunaan algoritma machine learning** yang mampu memodelkan hubungan yang kompleks antara fitur numerik dan kategorikal dengan harga perjalanan.
3.  **Evaluasi model** menggunakan metrik **Mean Squared Error (MSE)** untuk memastikan bahwa prediksi harga dapat dilakukan dengan error yang minimal.

Dengan model yang baik, perusahaan taksi dapat lebih mudah menyesuaikan tarif berdasarkan faktor-faktor variabel ini secara otomatis, meningkatkan kepuasan pelanggan, dan memastikan keuntungan yang lebih stabil bagi pengemudi.

Referensi : [ Predictive analysis of  taxi fare using  machine learning](https://scholar.googleusercontent.com/scholar?q=cache:AfAxrfc6xq8J:scholar.google.com/+Machine+Learning+for+Predicting+Taxi+Fare&hl=id&as_sdt=0,5)]

## Business Understanding

### Problem Statements

-  **Pernyataan Masalah 1**: Bagaimana cara memprediksi harga perjalanan taksi dengan akurat berdasarkan berbagai fitur seperti jarak, waktu, kondisi cuaca, dan tarif per kilometer?
-  **Pernyataan Masalah 2**: Bagaimana cara mengurangi error dalam prediksi harga dengan berbagai algoritma machine learning?

### Goals

-  **Jawaban Pernyataan Masalah 1**: Menggunakan model machine learning yang sesuai untuk memprediksi harga perjalanan taksi.
-  **Jawaban Pernyataan Masalah 2**: Mencapai model dengan error yang terkecil berdasarkan pengukuran Mean Squared Error (MSE).

### Solution Statements

-   **Solution 1**: Menggunakan algoritma Random Forest yang memiliki keunggulan dalam menangani dataset yang lebih besar dan memberikan hasil yang lebih baik pada data yang lebih kompleks.
-   **Solution 2**: Melakukan perbaikan pada model melalui tuning hyperparameter untuk mengoptimalkan hasil prediksi.

## Data Understanding

Dataset ini dirancang untuk memprediksi tarif perjalanan taksi berdasarkan berbagai faktor seperti jarak, waktu hari, kondisi lalu lintas, dan banyak lagi. Ini memberikan data sintetis yang realistis untuk tugas regresi, menawarkan peluang unik untuk mengeksplorasi tren penetapan harga di industri taksi.

Dataset : [Taxi Price Prediction](https://www.kaggle.com/datasets/denkuznetz/taxi-price-prediction)

## Exploratory Data Analysis - Deskripsi Variable

-   **Trip_Distance_km**: Jarak perjalanan dalam kilometer.
-   **Time_of_Day**: Waktu perjalanan dimulai (Morning, Afternoon, Evening, atau Night).
-   **Day_of_Week**: Indikasi perjalanan dilakukan pada weekday atau weekend
-   **Passenger_Count**: Jumlah penumpang.
-   **Traffic_Conditions**: Intensitas lalu lintas  (Low, Medium, High).
-   **Weather**: Kondisi cuaca (Clear, Rain, Snow).
-   **Base_Fare**: Tarif dasar perjalanan.
-   **Per_Km_Rate**: Tarif per kilometer.
-   **Per_Minute_Rate**: Tarif per menit perjalanan.
-   **Trip_Duration_Minutes**: Durasi total perjalanan dalam menit.
-   **Trip_Price**: Harga total perjalanan.

 Ada 1000 baris (records atau jumlah pengamatan) dalam dataset dan terdapat 11 kolom yaitu: Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions, Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes, Trip_Price.

### Kesalahan Tipe Data

![Screenshot 2025-02-02 115656](https://github.com/user-attachments/assets/b440f8e1-ac98-4e2c-a511-3c541205883a)

Terdapat 1 kolom yang memiliki tipe data tidak sesuai yaitu: Passenger_Count. Kolom Passenger_Count harus harus dikonversi dari float64 menjadi int64 karena jumlah penumpang adalah bilangan bulat.

### Missing Value

![image](https://github.com/user-attachments/assets/44d525d6-6a7e-41ee-a048-e8e53b8974f5)

Terdapat missing value di semua kolom kecuali Passenger_Count

### Duplicate Rows

![image](https://github.com/user-attachments/assets/acc3de55-2763-4d4d-a3d8-b417fc964b27)

Tidak ditemukan data duplikat

### Outliers

![image](https://github.com/user-attachments/assets/0a07821c-1600-4f71-99ea-c0c70db846cd)

Pada beberapa fitur numerik di atas terdapat outliers. Outliers tersebut akan diatasi dengan mengganti outlier dengan IQR.

## Exploratory Data Analysis - Univariate Analysis

### Categorical Features

![image](https://github.com/user-attachments/assets/4d6f9d60-3440-4951-b717-b3096ddf540e)

Data menunjukkan bahwa waktu sore (Afternoon) lebih dominan dibandingkan waktu lainnya. Ini bisa menunjukkan bahwa sebagian besar aktivitas perjalanan taxi terjadi pada sore hari.

![image](https://github.com/user-attachments/assets/73f59e88-dea9-413f-9223-1fe803c4b6b0)

Dapat disimpulkan sangat jarang terdapat penumpang yang menggunakan taxi pada saat weekend, yang mana hanya sekitar 29,7 %

![image](https://github.com/user-attachments/assets/a327c1c4-00ca-4dd7-a95e-64c1a43027e1)

Hampir tidak ada yang menggunakan taxi saat salju (snow), sebagian besar memilih cuaca yang terang.

### Numerical Features

![image](https://github.com/user-attachments/assets/1b8cdf11-a8ef-4a5c-bceb-34bec83f2ff1)

- Mayoritas perjalanan taxi memiliki jarak antara 0-20 km
- Jumlah penumpang relatif merata di antara nilai 1 hingga 4 penumpang.
- Tarif dasar mayoritas berkisar antara 2.0-5.0
- Harga perjalanan membentuk distribusi normal dengan sedikit right-skewed

## Exploratory Data Analysis - Multivariate Analysis

### Categorical Features

Mengecek rata-rata Trip_Price terhadap masing-masing fitur untuk mengetahui pengaruh fitur kategori terhadap Trip_Price.

![image](https://github.com/user-attachments/assets/f3824ef4-7054-4072-8697-bfc0decbc5f9)

Berdasarkan Time of Day (Waktu):
- Harga perjalanan cenderung konsisten di semua waktu (morning, afternoon, evening, night)
- Rata-rata harga berkisar antara 50-52
- Variasi harga antar waktu sangat kecil, menunjukkan tidak ada perbedaan tarif signifikan berdasarkan waktu
- Ini bisa menjadi indikasi bahwa perusahaan taxi tidak menerapkan surge pricing berdasarkan waktu

![image](https://github.com/user-attachments/assets/8c8a0779-4497-48df-b1aa-929fbd70bf70)

Berdasarkan Day of Week (Hari):
- Terdapat dua kategori: Weekday dan Weekend
- Rata-rata harga di weekday dan weekend hampir sama (sekitar 50-52)
- Tidak ada perbedaan pricing yang signifikan antara hari kerja dan akhir pekan
- Menunjukkan kebijakan harga yang konsisten sepanjang minggu

![image](https://github.com/user-attachments/assets/9c50e11d-6b66-40a2-8611-57fc34e61602)

Berdasarkan Traffic Conditions (Kondisi Lalu Lintas):
- Tiga kategori: Low, Medium, High
- Rata-rata harga relatif sama untuk semua kondisi lalu lintas (sekitar 50-52)
- Kondisi lalu lintas tidak mempengaruhi harga secara signifikan
- Ini menunjukkan bahwa tarif lebih didasarkan pada jarak dan waktu daripada kondisi lalu lintas

![image](https://github.com/user-attachments/assets/edff2729-0929-4511-ac1a-64b2287a07cc)
  
Berdasarkan Weather (Cuaca):
- Tiga kondisi cuaca: Clear, Rain, Snow
- Rata-rata harga konsisten di sekitar 50-52 untuk semua kondisi cuaca
- Tidak ada premium pricing untuk kondisi cuaca buruk
- Menunjukkan kebijakan harga yang fair tanpa mengambil keuntungan dari kondisi cuaca

### Numerical Features

![image](https://github.com/user-attachments/assets/cf874815-3d03-4ed3-b50b-eb93804267a3)

Pada pola sebaran data grafik pairplot, terlihat ‘Trip_Distance_km’, ‘Per_Km_Rate’, ‘Per_Minute_Rate’, dan ‘Trip_Duration_Minutes’ memiliki korelasi yang tinggi dengan fitur "Trip_Price". Sedangkan kedua fitur lainnya yaitu 'Passenger_Count' dan 'Base_Fare' terlihat memiliki korelasi yang lemah karena sebarannya tidak membentuk pola.

![image](https://github.com/user-attachments/assets/399b8c68-9702-4385-8456-5c4e85822ddb)

'Trip_Distance_km' memiliki korelasi positif yang kuat (0.68), sedangkan 'Passenger_Count' dan 'Base_Fare' memiliki korelasi yang sangat lemah. Sehingga, kedua fitur tersebut dapat di-drop

## Data Preparation

### Mengubah Tipe Data yang Tidak Sesuai

Kolom Passenger_Count harus diubah dari float64 menjadi int64 karena jumlah penumpang adalah bilangan bulat. Namun, kolom Passenger_Count mengandung nilai yang tidak dapat dikonversi menjadi integer, seperti NaN (nilai yang hilang) atau infinit. Maka, perlu menangani nilai yang hilang terlebih dahulu sebelum mengonversinya ke tipe int64.

```python
# Mengonversi kolom 'Passenger_Count' ke tipe integer
df['Passenger_Count'] = df['Passenger_Count'].astype('int64')
```
    
### Missing Value

Tahapan penanganan missing value untuk Atribut Kategorikal:

- Time_of_Day: Menggunakan modus (nilai yang paling sering muncul) berdasarkan pola waktu yang umum
- Day_of_Week: Menggunakan modus karena ini adalah data kategorikal dengan nilai tetap
- Traffic_Conditions: Menggunakan modus kondisi lalu lintas pada Time_of_Day yang sama
- Weather: Menggunakan modus cuaca pada hari dan waktu yang sama

```python
df['Time_of_Day'] = df['Time_of_Day'].fillna(df['Time_of_Day'].mode()[0])
df['Day_of_Week'] = df['Day_of_Week'].fillna(df['Day_of_Week'].mode()[0])
df['Weather'] = df.groupby('Day_of_Week')['Weather'].transform(lambda x: x.fillna(x.mode()[0] if  not x.mode().empty else  'Normal'))
df['Traffic_Conditions'] = df.groupby('Time_of_Day')['Traffic_Conditions'].transform(lambda x: x.fillna(x.mode()[0] if  not x.mode().empty else  'Normal')
```
  
Untuk Atribut Numerik:

- Trip_Distance_km: Menggunakan median untuk imputasi karena jarak perjalanan biasanya memiliki distribusi yang skewed
- Passenger_Count: Menggunakan modus karena ini adalah data diskrit dengan nilai yang terbatas
- Base_Fare, Per_Km_Rate, Per_Minute_Rate: Menggunakan median untuk menghindari pengaruh outlier
- Trip_Duration_Minutes: Menggunakan median karena durasi perjalanan juga cenderung memiliki distribusi yang skewed
- Trip_Price: Dihitung ulang berdasarkan Base_Fare + (Per_Km_Rate × Trip_Distance_km) + (Per_Minute_Rate × Trip_Duration_Minutes)

```python
# Mengimputasi missing value pada kolom numerikal
df['Passenger_Count'] = df['Passenger_Count'].fillna(df['Passenger_Count'].mode()[0])
df['Trip_Distance_km'] = df['Trip_Distance_km'].fillna(df['Trip_Distance_km'].median())
df['Trip_Duration_Minutes'] = df['Trip_Duration_Minutes'].fillna(df['Trip_Duration_Minutes'].median())
numeric_cols = ['Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate']
for col in numeric_cols:
  df[col] = df[col].fillna(df[col].median())
price_missing_mask = df['Trip_Price'].isna()
df.loc[price_missing_mask, 'Trip_Price'] = (df.loc[price_missing_mask, 'Base_Fare'] + 
 (df.loc[price_missing_mask, 'Per_Km_Rate'] * df.loc[price_missing_mask, 'Trip_Distance_km']) + 
  (df.loc[price_missing_mask, 'Per_Minute_Rate'] * df.loc[price_missing_mask, 'Trip_Duration_Minutes']))
```

### Menangani Outliers menggunakan IQR

```python
df_numerik = df.select_dtypes(include=['number'])

Q1 = df_numerik.quantile(0.25)
Q3 = df_numerik.quantile(0.75)
IQR=Q3-Q1
df=df[~((df_numerik<(Q1-1.5*IQR))|(df_numerik>(Q3+1.5*IQR))).any(axis=1)]

# Cek ukuran dataset setelah kita drop outliers
df.shape
```

### Encoding Fitur Kategorikal
Fitur kategorikal seperti 'Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', dan 'Weather' diubah menjadi bentuk numerik menggunakan teknik one-hot encoding.

### Train-Test-Split
Menggunakan proporsi pembagian sebesar 80:20 dengan fungsi train_test_split dari sklearn.

### Standarisasi
Data dinormalisasi dengan standar deviasi 1 dan rata-rata 0 menggunakan StandardScaler.

## Model Development

Pada proyek ini, digunakan tiga algoritma machine learning untuk memprediksi harga perjalanan taksi: **K-Nearest Neighbor (KNN)**, **Random Forest**, dan **Boosting (AdaBoost)**. Berikut adalah kelebihan dan kekurangan dari masing-masing algoritma yang digunakan:

1.  **K-Nearest Neighbor (KNN)**
    
    -   **Kelebihan**:
        -   **Mudah Dipahami**: KNN adalah algoritma yang sederhana dan mudah dipahami.
        -   **Tidak Membutuhkan Pemodelan**: KNN tidak memerlukan proses pelatihan yang rumit, karena data akan diproses secara langsung pada saat prediksi.
        -   **Kinerja Baik pada Data Kecil**: Algoritma ini dapat bekerja dengan baik pada dataset yang kecil dengan distribusi data yang tidak terlalu kompleks.
    -   **Kekurangan**:
        -   **Tidak Efisien untuk Dataset Besar**: KNN cenderung lambat pada dataset besar, karena harus memproses seluruh data pada saat prediksi.
        -   **Mudah Terpengaruh oleh Noise**: KNN sangat sensitif terhadap data yang berisik (noise) atau data yang memiliki banyak fitur yang tidak relevan.
        -   **Memerlukan Memori Besar**: Karena KNN menyimpan seluruh data, memori yang dibutuhkan untuk memproses data besar bisa sangat tinggi.
2.  **Random Forest**
    
    -   **Kelebihan**:
        -   **Menghindari Overfitting**: Random Forest menggunakan teknik ensemble learning yang melibatkan banyak pohon keputusan (decision trees), yang dapat menghindari overfitting pada data training.
        -   **Mampu Mengatasi Data yang Kompleks**: Dapat menangani hubungan non-linear dan data dengan banyak fitur.
        -   **Dapat Menangani Fitur Kategorikal dan Numerik**: Random Forest dapat menangani baik data kategorikal maupun numerik dengan efektif.
    -   **Kekurangan**:
        -   **Waktu Latih Lama pada Data Besar**: Proses pelatihan model bisa memakan waktu jika jumlah pohon keputusan yang digunakan sangat banyak.
        -   **Kurang Interpretabel**: Karena merupakan model berbasis ensemble, Random Forest cenderung kurang transparan dalam hal interpretasi hasil model jika dibandingkan dengan model yang lebih sederhana, seperti linear regression.
3.  **Boosting (AdaBoost)**
    
    -   **Kelebihan**:
        -   **Peningkatan Akurasi Model**: Boosting berusaha memperbaiki kesalahan model dengan menambahkan model-model tambahan yang difokuskan pada kesalahan sebelumnya, sehingga meningkatkan akurasi.
        -   **Meningkatkan Akurasi pada Model Lemah**: Biasanya, Boosting digunakan untuk meningkatkan kinerja model yang lemah (weak learners).
        -   **Cocok untuk Data yang Tidak Terlalu Banyak**: Meskipun ada risiko overfitting, model ini bekerja sangat baik pada data yang tidak terlalu besar atau yang memiliki pola yang tidak terlalu rumit.
    -   **Kekurangan**:
        -   **Overfitting pada Data yang Sangat Kompleks**: Jika tidak diatur dengan benar, Boosting dapat menyebabkan overfitting, terutama pada dataset yang lebih besar atau lebih rumit.
        -   **Pekerjaan Lambat pada Data Besar**: Proses pelatihan Boosting bisa lebih lambat jika dibandingkan dengan Random Forest, karena fokus pada kesalahan model sebelumnya.
     
#### Proses Improvement dengan Hyperparameter Tuning

Untuk meningkatkan kinerja model, dilakukan **hyperparameter tuning** pada algoritma **Random Forest**, karena algoritma ini menunjukkan hasil yang menjanjikan dalam evaluasi awal.

-   **Hyperparameter Tuning pada Random Forest**:
    -   **n_estimators**: Jumlah pohon yang digunakan dalam model. Semakin banyak pohon, semakin baik model dalam mengatasi variasi dalam data, tetapi juga semakin lama waktu pelatihan.
    -   **max_depth**: Menentukan kedalaman maksimum dari pohon keputusan. Pengaturan kedalaman yang tepat bisa mengurangi overfitting dan memastikan model tetap generalis.
    -   **min_samples_split**: Jumlah minimum sampel yang diperlukan untuk membagi sebuah node. Pengaturan ini mencegah pembagian yang terlalu mendalam yang dapat menyebabkan overfitting.
    -   **min_samples_leaf**: Jumlah minimum sampel yang diperlukan pada daun pohon keputusan untuk menghindari pembentukan pohon yang sangat dalam dan kompleks.

#### Memilih Model Terbaik

Berdasarkan evaluasi yang dilakukan dengan menggunakan **Mean Squared Error (MSE)** pada data pelatihan dan data uji, model **Random Forest** menunjukkan hasil yang paling baik dengan error yang terkecil. Berikut adalah hasil perbandingan **MSE** dari masing-masing model:

![image](https://github.com/user-attachments/assets/dff611f7-92b7-4cc7-b1f5-3a2c1294d535)


**Mengapa Memilih Random Forest sebagai Model Terbaik**:

1.  **Error Terendah**: Random Forest memberikan error yang lebih rendah dibandingkan dengan model lainnya pada data uji.
2.  **Stabilitas**: Random Forest menunjukkan kinerja yang lebih konsisten, tidak terpengaruh oleh variasi data.
3.  **Kemampuan untuk Mengatasi Data yang Kompleks**: Dengan teknik ensemble, Random Forest mampu mengatasi hubungan non-linear dan kompleks antara fitur.

## Model Evaluation

Evaluasi model dilakukan dengan menggunakan metrik Mean Squared Error (MSE) untuk membandingkan hasil prediksi antara model yang berbeda. Berdasarkan hasil pengujian, model Random Forest memberikan error yang paling kecil dibandingkan dengan model KNN dan Boosting. 

Formula MSE adalah sebagai berikut:

![image](https://github.com/user-attachments/assets/d873656c-8116-4837-ae5f-c73d8dbb1286)

**Penjelasan Formula**:

-   n adalah jumlah sampel atau data pada dataset.
-   yi adalah nilai sebenarnya (true value) dari data ke-i (nilai target yang sebenarnya, dalam hal ini harga perjalanan taksi yang sebenarnya).
-   y^i adalah nilai prediksi (predicted value) yang dihasilkan oleh model untuk data ke-i.
-   (yi−y^i) adalah selisih antara nilai sebenarnya dan nilai prediksi, yang dikenal dengan istilah **residual** atau **error**.
-   (yi−y^i)^2 adalah kuadrat dari selisih tersebut, yang berfungsi untuk memastikan bahwa error yang positif dan negatif tidak saling menghilangkan. Juga, memberikan bobot lebih besar pada error yang lebih besar.
-   ∑i=1n berarti menjumlahkan kuadrat error untuk semua data dalam dataset.
-   1/n adalah rata-rata dari kuadrat error, untuk mendapatkan nilai error yang lebih representatif di seluruh dataset.

MSE digunakan untuk mengukur seberapa baik model prediksi dalam menghasilkan nilai yang mendekati nilai sebenarnya. Semakin kecil nilai MSE, semakin baik model dalam melakukan prediksi. Metrik ini mengukur rata-rata kuadrat selisih antara nilai yang diprediksi oleh model dengan nilai yang sebenarnya.

- MSE yang Kecil: Menandakan bahwa prediksi model mendekati nilai yang sebenarnya, yang menunjukkan bahwa model mampu mempelajari pola dengan baik dari data dan melakukan prediksi yang akurat.
- MSE yang Besar: Menunjukkan bahwa model menghasilkan prediksi yang jauh dari nilai yang sebenarnya, yang mengindikasikan bahwa model kurang baik dalam menangani data dan menghasilkan estimasi yang akurat.

Misalnya, dalam konteks proyek ini, kita memiliki harga perjalanan taksi yang sebenarnya (𝑦𝑖) dan harga yang diprediksi oleh model (𝑦^𝑖) untuk setiap perjalanan. MSE mengukur sejauh mana prediksi harga perjalanan yang dihasilkan oleh model berbeda dari harga yang sebenarnya.


