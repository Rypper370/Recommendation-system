# Laporan Proyek Machine Learning - Yohanes Aldo Anantha

## Project Overview

Dalam era digital yang sarat dengan informasi, pengguna sering kali dihadapkan pada tantangan dalam memilih buku yang sesuai dengan minat dan preferensi mereka. Sistem rekomendasi buku menjadi solusi penting untuk menyaring informasi dan memberikan saran yang relevan kepada pengguna. Proyek ini bertujuan untuk mengembangkan sistem rekomendasi buku yang efektif dengan memanfaatkan dua pendekatan utama: content-based filtering dan collaborative filtering. Content-based filtering menganalisis fitur-fitur buku seperti penulis dan rating untuk merekomendasikan buku yang mirip, sementara collaborative filtering memanfaatkan pola interaksi pengguna untuk memberikan rekomendasi yang dipersonalisasi. Dengan membandingkan dua pendekatan ini secara terpisah, proyek ini bertujuan untuk mengevaluasi kualitas dan efektivitas masing-masing model dalam memberikan rekomendasi yang akurat dan relevan  ([1](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00592-5)). Pendekatan ini juga memungkinkan identifikasi kelebihan dan kekurangan dari setiap metode, khususnya dalam menghadapi tantangan umum seperti cold start dan data sparsity pada sistem rekomendasi tradisional. Pengembangan sistem ini didasarkan pada dataset "Best Books of the Decade: 2020's" dari Kaggle, yang menyediakan data buku dan ulasan pengguna yang kaya untuk analisis dan pelatihan model.

### Referensi 
[1] [S. K. Behera, A. Kumar, and D. P. Mohapatra, "A systematic review and research perspective on recommender systems: Techniques, challenges, and future directions," Journal of Big Data, vol. 9, no. 1, p. 43, 2022. [Online]. Available: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00592-5]


## Business Understanding

Di tengah maraknya terbitan buku baru setiap tahunnya, pengguna sering mengalami kesulitan dalam menemukan buku yang sesuai dengan minat mereka. Hal ini menciptakan kebutuhan akan sistem rekomendasi yang dapat menyajikan pilihan bacaan yang relevan dan personal. Proyek ini bertujuan untuk membangun model rekomendasi buku yang mampu memahami preferensi pengguna melalui pendekatan content-based dan collaborative filtering. Dengan sistem ini, diharapkan proses pencarian buku menjadi lebih efisien sekaligus meningkatkan kepuasan membaca. Selain itu, sistem rekomendasi juga dapat menjadi alat strategis bagi penulis dan penerbit untuk menjangkau pembaca yang lebih tepat sasaran.

### Problem Statements

- Bagaimana cara membantu pengguna menemukan buku baru yang sesuai dengan preferensi bacaan mereka di antara ribuan buku yang terbit di dekade 2020-an?
- Bagaimana cara membangun model machine learning yang efektif untuk memberikan rekomendasi buku yang dipersonalisasi berdasarkan riwayat bacaan pengguna atau kemiripan antar buku?
- Bagaimana cara mengevaluasi dan membandingkan performa model Content-Based Filtering dan Collaborative Filtering untuk menentukan pendekatan yang paling efektif dalam memberikan rekomendasi buku?

### Goals
- Mengembangkan sistem rekomendasi buku menggunakan metode Content-Based Filtering yang merekomendasikan buku berdasarkan kemiripan konten (misalnya, penulis atau judul).
- Mengembangkan sistem rekomendasi buku menggunakan metode Collaborative Filtering yang merekomendasikan buku berdasarkan pola preferensi pengguna serupa.
- Mengevaluasi performa kedua model rekomendasi yang dibangun untuk menentukan pendekatan mana yang lebih sesuai.

### Solution statements
- Content-Based Filtering = pendekatan content-based filtering digunakan untuk merekomendasikan buku berdasarkan kesamaan fitur kontennya. Dalam implementasinya, sistem menggabungkan informasi dari kolom Author dan Rating sebagai fitur utama yang merepresentasikan setiap buku. Fitur gabungan ini kemudian diubah ke dalam bentuk numerik menggunakan metode TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. Setelah fitur buku direpresentasikan dalam bentuk vektor, sistem menghitung tingkat kemiripan antar buku dengan cosine similarity.
- Collaborative filtering dalam proyek ini diimplementasikan menggunakan pendekatan neural network dengan model matrix factorization berbasis embedding. Sistem menggunakan data interaksi pengguna terhadap buku berupa userId, bookIndex, dan skor Rating. Baik ID pengguna maupun ID buku dipetakan ke dalam vektor laten melalui embedding layer, yang kemudian digabungkan dan diteruskan ke beberapa lapisan dense untuk memprediksi rating. Model dilatih dengan optimasi menggunakan Mean Squared Error (MSE) loss dan Adam optimizer. Untuk meningkatkan kinerja model, digunakan teknik Early Stopping dan ReduceLROnPlateau agar model tidak overfit dan tetap adaptif selama proses pelatihan.

## Data Understanding
Dataset yang digunakan berjudul ([Best Books of the Decade: 2020’s dan bersumber dari Kaggle](https://www.kaggle.com/datasets/valakhorasani/best-books-of-the-decade-2020s)), disusun oleh Vala Khorasani. Dataset ini terdiri dari dua bagian utama: data buku dan data ulasan pengguna. Bagian pertama mencakup informasi tentang 2.327 buku terbaik dekade 2020-an, termasuk judul, penulis, rating, jumlah suara, dan skor agregat. Bagian kedua berisi sekitar 600.000 ulasan pengguna yang mengaitkan skor dengan buku tertentu melalui indeks buku. Data ini sangat mendukung pengembangan sistem rekomendasi berbasis konten maupun perilaku pengguna.
1.  **Best Books of the Decade: 2020s**: Bagian ini merupakan kumpulan data dari 2.327 buku terbaik yang diterbitkan selama dekade 2020-an. Pemilihan buku didasarkan pada peringkat (rating) pengguna dan popularitasnya. Informasi yang tersedia untuk setiap buku meliputi:
    *   `Index`: Pengenal unik untuk setiap buku.
    *   `Book Name`: Judul buku.
    *   `Author`: Nama penulis buku.
    *   `Rating`: Rata-rata peringkat yang diberikan oleh pengguna (dalam skala 1 hingga 5).
    *   `Number of Votes`: Jumlah total suara atau ulasan yang diterima buku tersebut.
    *   `Score`: Skor agregat yang dihitung berdasarkan kombinasi peringkat dan jumlah suara.

2.  **User Reviews**: Bagian kedua ini berisi data ulasan yang dibuat oleh pengguna, dengan total mencapai 600.000 ulasan. Data ulasan ini sangat berguna untuk analisis preferensi pengguna dan pengembangan sistem rekomendasi berbasis collaborative filtering. Kolom yang tersedia dalam bagian ini adalah:
    *   `userId`: Pengenal unik untuk setiap pengguna yang memberikan ulasan.
    *   `bookIndex`: Indeks buku yang merujuk pada `Index` di bagian pertama dataset, menghubungkan ulasan dengan buku spesifik.
    *   `score`: Skor atau peringkat yang diberikan oleh pengguna untuk buku tersebut (dalam skala 1 hingga 5).


### Informasi Data Buku
```bash
Data columns (total 6 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   Index            2329 non-null   int64 
 1   Book Name        2329 non-null   object
 2   Author           2329 non-null   object
 3   Rating           2329 non-null   object
 4   Number of Votes  2329 non-null   object
 5   Score            2329 non-null   int64 
dtypes: int64(2), object(4)
memory usage: 109.3+ KB

Mengecek Missing Value : 
Index              0
Book Name          0
Author             0
Rating             0
Number of Votes    0
Score              0
dtype: int64

Mengecek Duplicate : 0

Mengecek nilai unique : 
Index              2329
Book Name          2327
Author             1768
Rating              176
Number of Votes    2212
Score               634
```
Berdasarkan informasi analisis diatas, dataset `book_df` berisi **2329** buku dengan 6 kolom data (**Index**, **Book Name**, **Author**, **Rating**, **Number of Votes**, **Score**). Dataset ini sangat bersih karena tidak ada missing values sama sekali di semua kolom. Terdapat variasi yang baik dalam data dengan **2327** judul buku unik, **1768** penulis unik, dan **176** rating berbeda. Kolom **Number of Votes** memiliki **2212** nilai unik dan **Score** memiliki **634** nilai unik, menunjukkan distribusi data yang beragam.

### Informasi Data User
```bash
Data columns (total 3 columns):
 #   Column     Non-Null Count   Dtype
---  ------     --------------   -----
 0   userId     600000 non-null  int64
 1   bookIndex  600000 non-null  int64
 2   score      600000 non-null  int64
dtypes: int64(3)
memory usage: 13.7 MB

Mengecek Missing Value : 
userId       0
bookIndex    0
score        0
dtype: int64

Mengecek Duplicate : 175

Mengecek nilai unique : 
userId       79957
bookIndex     2327
score            5
```
Berdasarkan informasi diatas, data ini berisi **600.000** entri dengan 3 kolom data **(userId, bookIndex, score)** yang semuanya bertipe integer. Dataset ini sangat bersih karena tidak ada missing values sama sekali di semua kolom. Terdapat **79.957 pengguna unik**, **2.327 buku unik**, namun hanya **5 nilai score** unik yang menunjukkan sistem rating terbatas. Dataset memiliki **175 baris duplikat** yang perlu diperhatikan untuk analisis lebih lanjut.

### Visualisasi  Data 
![image](https://github.com/user-attachments/assets/5400f35f-f102-4255-ab10-def5235751e2)
Berdasarkan hasil analisis grafik diatas, menampilkan 10 buku dengan rating tertinggi berdasarkan rata-rata penilaian pengguna dalam dataset sistem rekomendasi. Buku **Anshuman krit Saral Vastu Gyan** dan **Metaphysics of Sound** menduduki peringkat teratas dengan rating mendekati **5.0**, menunjukkan kepuasan pengguna yang sangat tinggi terhadap kedua buku tersebut. Sebagian besar buku dalam daftar ini memiliki rating di atas **4.5**, yang mengindikasikan kualitas konten yang baik dan respon positif dari pembaca. Informasi ini memberikan gambaran tentang preferensi umum pengguna dan dapat menjadi acuan untuk memahami jenis buku apa yang cenderung mendapat apresiasi tinggi dalam dataset ini.

![image](https://github.com/user-attachments/assets/8e857ae6-f4ad-48ec-8cc0-6f8ab8a373f8)
Berdasarkan hasil analisis grafik diatas, menampilkan 10 penulis dengan rating tertinggi berdasarkan rata-rata penilaian dari semua buku yang mereka tulis dalam dataset sistem rekomendasi. **Anshuman Srivastav** dan **Nataša Pantović** menduduki peringkat teratas dengan rating mendekati **5.0**, menunjukkan bahwa karya-karya mereka secara konsisten mendapat apresiasi tinggi dari pembaca. Sebagian besar penulis dalam daftar ini memiliki rating di atas **4.5**, yang mengindikasikan kualitas tulisan yang baik dan kemampuan mereka dalam menciptakan konten yang memuaskan pembaca.

![image](https://github.com/user-attachments/assets/c70d9f0e-315e-439d-81ff-31277afecb34)
Berdasarkan hasil analisis grafik diatas, menampilkan distribusi jumlah buku yang diulas per pengguna dalam dataset sistem rekomendasi, dengan menggunakan skala logaritmik pada sumbu y untuk mengakomodasi rentang data yang luas. Sebagian besar pengguna memberikan ulasan untuk **6-8 buku**, yang ditunjukkan dengan puncak distribusi pada rentang tersebut dengan jumlah pengguna mencapai lebih dari **10.000 orang**. Terdapat penurunan signifikan jumlah pengguna seiring dengan bertambahnya jumlah buku yang diulas, menunjukkan bahwa hanya sedikit pengguna yang sangat aktif dalam memberikan banyak ulasan. Distribution ini mencerminkan pola umum dalam platform review dimana mayoritas pengguna bersifat casual dengan aktivitas terbatas, sementara pengguna yang memberikan **15+ ulasan** sangat sedikit jumlahnya. Pada ujung ekstrim, terdapat pengguna yang memberikan hingga **20+ ulasan** dengan jumlah yang sangat kecil (di bawah 10 pengguna).

![image](https://github.com/user-attachments/assets/22ca93d9-c354-49b4-8f9e-5e3a0a242507)

Berdasarkan hasil analisis grafik diatas, menampilkan bahwa pola penilaian menunjukkan sebaran yang relatif merata di seluruh kategori rating dari 1 hingga 5. **Rating 1 (sangat tidak puas)** memiliki **frekuensi tertinggi** dengan **120.498 ulasan**, menunjukkan adanya sejumlah besar pengguna yang memberikan penilaian **negatif**. **Rating 5 (sangat puas)** berada di posisi kedua dengan **120.223 ulasan**, diikuti oleh **rating 2** dengan **120.109 ulasan**, yang mengindikasikan polarisasi dalam kepuasan pengguna. **Rating 3 dan 4** memiliki frekuensi yang hampir sama dengan masing-masing **119.507** dan **119.488 ulasan**, menunjukkan konsistensi dalam penilaian **netral** hingga **positif**. Distribusi yang **seimbang** ini dengan variasi hanya sekitar **1.000 ulasan** antar kategori menunjukkan bahwa dataset ini cukup representatif dan tidak memiliki bias yang signifikan, sehingga dapat memberikan gambaran yang objektif tentang tingkat kepuasan pengguna secara keseluruhan.

## Data Preparation
Tahap persiapan data dilakukan untuk memastikan kualitas data sebelum digunakan dalam pemodelan. Proses ini mencakup pembersihan data, transformasi fitur, dan penanganan nilai yang hilang agar model dapat belajar dengan optimal. 
### Book_df 
Dataset `book_df` memerlukan beberapa penyesuaian kunci agar siap digunakan, khususnya untuk model Content-Based Filtering.
* Pembersihan dan Konversi Tipe Data Kolom `Rating `dan` Number of Votes`:  merupakan langkah awal melibatkan pembersihan kolom Rating dan Number of Votes yang awalnya bertipe `object`. Menggunakan `pd.to_numeric` dengan `errors='coerce'`, nilai non-numerik dikonversi menjadi `NaN`. Baris yang mengandung `NaN` pada kolom-kolom vital ini kemudian dihapus melalui `.dropna()` untuk menjaga integritas data. Kolom Rating selanjutnya dipastikan bertipe `float` untuk konsistensi.
* Pembuatan Fitur Gabungan (features) untuk Content-Based Filtering: Untuk keperluan Content-Based Filtering, dibuat kolom fitur gabungan bernama `features`. Ini dilakukan dengan mengkombinasikan informasi tekstual dari kolom `Author` dan `Rating` (yang telah dikonversi ke string), dipisahkan oleh spasi. Fitur gabungan ini menjadi dasar representasi konten buku.
* Ekstraksi Fitur TF-IDF: Walaupun TF-IDF diterapkan pada tahap Modeling, teknik TF-IDF merupakan bagian penting dari persiapan fitur konten. TF-IDF akan mengubah kolom `features` tekstual menjadi representasi vektor numerik, memungkinkan perhitungan kemiripan antar buku berdasarkan konten penulis dan rating.

### User_df
Dataset `user_df`, yang menjadi fondasi model Collaborative Filtering, melalui serangkaian persiapan yang berbeda.
* Menghapus Duplikat: Langkah pertama adalah menghapus baris duplikat dari user_df menggunakan `.drop_duplicates()`. Ini memastikan bahwa setiap interaksi unik pengguna-buku-rating hanya tercatat sekali, menjaga integritas data. Berdasarkan eksekusi notebook, sebanyak 175 baris duplikat berhasil dihapus.
* Penggabungan Data (merge): Untuk memperkaya data interaksi, `user_df` digabungkan `(pd.merge)` dengan informasi Book Name dari `book_df.` Penggabungan dilakukan berdasarkan `bookIndex` pada `user_df`  dan `Index` pada `book_df` menggunakan `left join`.
* Mapping ID Pengguna dan Buku ke Indeks Berurutan: Langkah krusial berikutnya adalah memetakan `userId` dan `bookIndex` asli ke indeks integer yang berurutan (mulai dari 0). Ini dilakukan dengan membuat dictionary pemetaan `(user_to_index, book_to_index)` dari ID unik dan menerapkannya menggunakan metode `.map()`, menghasilkan kolom baru `userId_mapped` dan `bookIndex_mapped`. Pemetaan ini penting untuk efisiensi input model, terutama layer Embedding.
```python
# Melakukan Mapping user dan book ke indices
user_ids = merged_df['userId'].unique().tolist()
book_indices = merged_df['bookIndex'].unique().tolist()
user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
book_to_index = {book_index: index for index, book_index in enumerate(book_indices)}

# Melakukan reverse mapping untuk analisis nanti
index_to_user = {index: user_id for user_id, index in user_to_index.items()}
index_to_book = {index: book_index for book_index, index in book_to_index.items()}

# mapping
merged_df['userId_mapped'] = merged_df['userId'].map(user_to_index)
merged_df['bookIndex_mapped'] = merged_df['bookIndex'].map(book_to_index)
```
* Penyusunan Fitur (X) dan Target (y): Data disusun menjadi format standar untuk training model. Kolom hasil mapping `(userId_mapped, bookIndex_mapped)` dijadikan matriks fitur input (X), sementara kolom score (rating asli) dijadikan vektor target output (y) dan dikonversi ke tipe `float32`.
* Data Splitting (Train/Validation Split) : Dataset dibagi menjadi set pelatihan (80%) dan validasi (20%) menggunakan train_test_split dari `sklearn.model_selection`. Penggunaan random_state=42 memastikan pembagian yang acak namun dapat direproduksi, memungkinkan evaluasi model yang objektif.
* Normalisasi Target (y): Nilai target (y) dinormalisasi menggunakan `Z-score`. Rata-rata (train_mean) dan standar deviasi (train_std) dihitung hanya dari data pelatihan (y_train) untuk mencegah data leakage. Normalisasi kemudian diterapkan pada y_train dan y_val untuk menghasilkan y_train_norm dan y_val_norm, yang membantu konvergensi model.

## Modeling
### Content-Based Filtering
Content-Based Filtering merekomendasikan buku berdasarkan kemiripan fitur konten antar buku berdasarkan `Author` dan `Rating`. Model ini tidak memiliki arsitektur berlapis seperti Neural Network, melainkan terdiri dari serangkaian komponen pemrosesan dan perhitungan.
* Komponen Utama Model
   * Ekstraksi Fitur dengan TF-IDF: Komponen ini bertanggung jawab mengubah fitur tekstual gabungan (Author dan Rating) menjadi representasi vektor numerik. TF-IDF mengukur pentingnya setiap kata dalam konteks seluruh dataset buku.
      * Parameter: `stop_words=\'english\'`, `max_features=1000`.
      * Output: Matriks TF-IDF (setiap baris adalah vektor buku).
* Perhitungan Kemiripan dengan Cosine Similarity: Komponen ini mengambil matriks TF-IDF sebagai input dan menghitung kemiripan antar semua pasangan vektor buku menggunakan metrik Cosine Similarity.
     * Output: Matriks Cosine Similarity (n×n), di mana nilai sel (i, j) menunjukkan kemiripan antara buku i dan buku j.
```python
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf.fit_transform(book_df['features'])
cosine_sim = cosine_similarity(tfidf_matrix)

def get_content_recommendations(book_index, num_rec=5):
    sim_scores = list(enumerate(cosine_sim[book_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_rec+1]

    # Mendapatkan buku dari indices dan similarity scores
    book_indices = [i[0] for i in sim_scores]
    similarities = [i[1] for i in sim_scores]

    #Membuat hasil dengan cosine similarity
    result = book_df.iloc[book_indices][['Book Name', 'Author', 'Rating']].copy()
    result['Cosine_Similarity'] = [f"{score:.4f}" for score in similarities]

    return result
```
* Cara Kerja Model Berdasarkan Hasil
Hasil rekomendasi untuk buku "The Invisible Life of Addie LaRue" (Victoria Schwab, Rating 4.18) menunjukkan bagaimana model bekerja:
```bash
==================================================
REKOMENDASI UNTUK BUKU INDEX 0
==================================================
Target Book: The Invisible Life of Addie LaRue
Author: Victoria Schwab
Rating: 4.18

Top 5 Recommendations:
                                          Book Name            Author  Rating Cosine_Similarity
The Fragile Threads of Power (Threads of Power, #1)   Victoria Schwab    4.30            0.8723
                Bridge of Souls (Cassidy Blake, #3)   Victoria Schwab    4.03            0.7668
                                            Gallant   Victoria Schwab    3.71            0.7518
                                              China Edward Rutherfurd    4.18            0.4890
                  Realm Breaker (Realm Breaker, #1)  Victoria Aveyard    3.59            0.4398
```
   * Pengaruh Penulis: Tiga rekomendasi teratas adalah buku dari penulis yang sama **Victoria Schwab* dengan nilai similarity tinggi `(0.75-0.87)`, menunjukkan bahwa model memberikan bobot besar pada kesamaan penulis.
   * Pengaruh Rating: Buku **China* oleh **Edward Rutherfurd* memiliki rating persis sama `(4.18)` dengan buku target, menjelaskan mengapa muncul di posisi keempat meskipun penulisnya berbeda.
   * Kemiripan Nama Penulis: **Victoria Aveyard* muncul di posisi kelima, kemungkinan karena kemiripan nama depan dengan penulis target.
   * Pola Penurunan Similarity: Terlihat penurunan drastis nilai similarity `(dari 0.75 ke 0.48)` saat beralih ke penulis berbeda, mengkonfirmasi dominasi fitur penulis dalam model ini.

* Kelebihan dan Kekurangan Model Content-Based Filtering
   * Kelebihan:
      * Tidak Memerlukan Data Pengguna Lain: Model dapat memberikan rekomendasi bahkan untuk pengguna baru yang belum memiliki riwayat interaksi (mengatasi masalah cold start).
      * Transparansi Tinggi: Hasil rekomendasi mudah dijelaskan berdasarkan fitur yang digunakan (penulis dan rating), meningkatkan kepercayaan pengguna.
      * Spesifisitas Tinggi: Sangat efektif menemukan buku dari penulis yang sama atau dengan karakteristik serupa.
      * Komputasi Efisien: Setelah matriks similarity dihitung, rekomendasi dapat dihasilkan dengan cepat tanpa perlu pelatihan ulang.
   
   * Kekurangan:
      * Keterbatasan Fitur: Hanya menggunakan penulis dan rating, tidak menangkap aspek penting lain seperti genre, tema, atau gaya penulisan.
      * Overspecialization: Cenderung merekomendasikan buku yang sangat mirip (terutama dari penulis yang sama), kurang memberikan variasi atau penemuan baru.
      * Ketergantungan pada Metadata: Kualitas rekomendasi sangat bergantung pada kelengkapan dan akurasi metadata buku.
      * Tidak Mempelajari Preferensi Pengguna: Tidak mempertimbangkan pola preferensi pengguna atau popularitas buku di kalangan pengguna serupa.

### Collaborative Filtering
Collaborative Filtering merekomendasikan buku berdasarkan pola interaksi pengguna-buku, dengan asumsi bahwa pengguna yang memiliki preferensi serupa di masa lalu akan memiliki preferensi serupa di masa depan. Model ini diimplementasikan menggunakan Neural Network dengan pendekatan matrix factorization berbasis embedding.
* Arsitektur Model
   * Input Layer: Dua input terpisah untuk userId_mapped dan bookIndex_mapped
   * Embedding Layer: Masing-masing input diproses oleh layer embedding terpisah dengan dimensi 50, menghasilkan representasi vektor laten untuk setiap pengguna dan buku
   * Flatten Layer: Mengubah output embedding dari bentuk 3D menjadi vektor 1D
   * Concatenate Layer: Menggabungkan vektor pengguna dan buku menjadi satu vektor gabungan
   * Dense Layers: Dua layer dense (128 dan 64 unit) dengan aktivasi ReLU untuk mempelajari interaksi non-linear
   * Dropout Layers: Dropout dengan rate 0.5 setelah setiap dense layer untuk mencegah overfitting
   * Output Layer: Layer dense dengan 1 unit dan aktivasi linear untuk memprediksi rating

```python
def build_model(self):
        #Membangun arsitektur model

        # Input layers
        user_input = Input(shape=(), name='user_id')
        book_input = Input(shape=(), name='book_id')

        # Embedding layers - mengubah ID menjadi vector
        user_embedding = Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            name='user_embedding'
        )(user_input)

        book_embedding = Embedding(
            input_dim=self.num_books,
            output_dim=self.embedding_dim,
            name='book_embedding'
        )(book_input)

        # Flatten embedding menjadi 1D vector
        user_vec = Flatten()(user_embedding)
        book_vec = Flatten()(book_embedding)

        # Gabungkan user dan book vectors
        concat = Concatenate()([user_vec, book_vec])

        # Dense layers untuk pembelajaran pola yang lebih kompleks
        dense1 = Dense(128, activation='relu')(concat)
        dropout1 = Dropout(0.2)(dense1)

        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)

        # Output layer - prediksi rating/score
        output = Dense(1, activation='linear')(dropout2)

        # Compile model
        model = Model(inputs=[user_input, book_input], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=[RootMeanSquaredError(name='rmse'), 'mae']
        )

        self.model = model
        return model
``` 

* Parameter Training
   * Optimizer: Adam dengan learning rate awal 0.001
   * Loss Function: Mean Squared Error (MSE)
   * Metrics: Root Mean Squared Error (RMSE) dan Mean Absolute Error (MAE)
   * Epochs: 10
   * Batch Size: 256
   * Callbacks:
      * `ReduceLROnPlateau`: Mengurangi learning rate dengan faktor 0.5 jika validation loss tidak membaik selama 2 epoch
      * `EarlyStopping`: Menghentikan training jika validation loss tidak membaik selama 5 epoch dan mengembalikan bobot terbaik

* Hasil Training dan Cara Kerja Model
Hasil training pada epoch terakhir:
```bash
Epoch 10/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 86s 46ms/step - loss: 0.1261 - mae: 0.2761 - rmse: 0.3551 - val_loss: 1.4836 - val_mae: 1.0078 - val_rmse: 1.2180 - learning_rate: 5.0000e-04
```
   * Proses Pembelajaran: Model belajar dengan baik pada data training `(RMSE 0.3551)`, tetapi performa pada data validasi jauh lebih buruk `(RMSE 1.2180)`, menunjukkan adanya overfitting
   * Adaptasi Learning Rate: Learning rate diturunkan menjadi 0.0005 pada epoch terakhir, menunjukkan callback `ReduceLROnPlateau` aktif karena performa validasi tidak membaik

Contoh hasil rekomendasi untuk User `65674`:

```bash
User ini pernah rating 7 buku
Buku favorit user (Contoh):
  - Camp Jupiter Classified: A Probatios Jo (Rating: 5)
  - Every Missing Piece (Rating: 4)
  - Eden Mine (Rating: 3)

Rekomendasi buku (Prediksi Skor):
  1. Lessons in Chemistry (Score: 3.03)
  2. Viviana Valentine Gets Her Man (Girl Fri (Score: 3.03)
  3. Birds: Photographs (Score: 3.03)
  4. Difficult Women: A History of Feminism i (Score: 3.03)
  5. My Dark Vanessa (Score: 3.02)
```
   * Pola Prediksi: Model cenderung memberikan prediksi skor yang konservatif (sekitar 3.0) untuk semua buku yang direkomendasikan, meskipun user memiliki riwayat rating tinggi (5) untuk beberapa buku
   * Kurangnya Diferensiasi: Perbedaan skor antar rekomendasi sangat kecil (3.03 vs 3.02), menunjukkan model kesulitan membedakan preferensi secara signifikan
   * Konsistensi Rekomendasi: Buku **Lessons in Chemistry* dan **Viviana Valentine Gets Her Man* muncul sebagai rekomendasi teratas untuk beberapa user berbeda, menunjukkan model mungkin lebih memperhitungkan popularitas buku daripada preferensi personal
   * Pengaruh Embedding: Meskipun model menggunakan embedding untuk mempelajari representasi laten pengguna dan buku, hasil menunjukkan bahwa representasi ini belum cukup kuat untuk menghasilkan rekomendasi yang sangat personal
* Kelebihan dan Kekurangan Model Collaborative Filtering (Neural Network)
   * Kelebihan:
      * Potensi Personalisasi Tinggi: Mampu mempelajari pola preferensi kompleks dari interaksi pengguna, berpotensi memberikan rekomendasi yang lebih personal
      * Kemampuan Menemukan Rekomendasi Serendipitous: Dapat merekomendasikan item yang tidak terduga atau di luar kebiasaan pengguna berdasarkan pola pengguna lain
      * Tidak Bergantung pada Metadata: Tidak memerlukan fitur konten buku yang detail, hanya data interaksi
   
   * Kekurangan:
      * Rentan Overfitting: Seperti terlihat pada hasil validasi (RMSE tinggi), model cenderung menghafal data training dan kurang generalisasi
      * Masalah Cold Start: Sulit memberikan rekomendasi untuk pengguna baru atau buku baru yang belum memiliki data interaksi
      * Membutuhkan Data Interaksi yang Banyak: Kinerja model sangat bergantung pada ketersediaan data rating/interaksi yang cukup
      * Prediksi Kurang Terdiferensiasi (Saat Ini): Hasil prediksi skor cenderung konservatif dan kurang bervariasi, mengurangi personalisasi yang dirasakan

## Evaluation
### Evaluation Content-Based Filtering
   * Metrik Evaluasi yang Digunakan: Mengingat model Content-Based Filtering ini bertujuan menemukan buku yang mirip berdasarkan fitur konten (Author, Rating) dan tidak memprediksi nilai rating spesifik, metrik evaluasi standar seperti RMSE/MAE tidak relevan. Oleh karena itu, evaluasi utama menggunakan analisis kualitatif terhadap relevansi, diversitas, dan karakteristik rekomendasi yang dihasilkan untuk sampel buku.
   * Kesesuaian Metrik: Analisis kualitatif sesuai karena problem statement untuk CBF adalah menemukan item serupa. Metrik ini memungkinkan penilaian langsung terhadap kualitas kemiripan yang ditemukan (apakah rekomendasi masuk akal berdasarkan fitur yang digunakan?) dan karakteristik output model (apakah cenderung overspecialized?).
   * Cara Kerja Metrik (Analisis Kualitatif): Proses ini melibatkan pemilihan buku target secara acak atau representatif, menjalankan fungsi rekomendasi untuk mendapatkan N buku teratas, lalu menganalisis daftar rekomendasi tersebut berdasarkan:
      * Relevansi: Apakah buku yang direkomendasikan secara logis terkait dengan buku target berdasarkan fitur yang digunakan (misal, penulis yang sama)?
      * Kualitas: Apakah buku yang direkomendasikan memiliki rating yang baik atau sesuai dengan ekspektasi?
      * Diversitas: Seberapa beragam rekomendasi yang diberikan (misal, berapa banyak penulis berbeda)?
```bash
EVALUASI DETAIL UNTUK SATU BUKU:

Target Book: The House in the Cerulean Sea (Cerulean Chronicles, #1)
Target Rating: 4.4
Target Author: T.J. Klune

Rekomendasi yang diberikan:
                             Book Name           Author  Rating   Cosine_Similarity
21           Under the Whispering Door       T.J. Klune    4.15         0.8187
230            In the Lives of Puppets       T.J. Klune    3.92         0.8037 
0    The Invisible Life of Addie LaRue  Victoria Schwab    4.18         0.0000  
2                    Project Hail Mary        Andy Weir    4.51         0.0000
3                 The Midnight Library        Matt Haig    3.99         0.0000
HASIL EVALUASI:
Rata-rata rating rekomendasi: 4.20
Jumlah rekomendasi rating >= 4.0: 3/5
Author diversity: 0.80 (4 author berbeda)
Rata-rata cosine similarity: 0.3245
{'avg_rating': 4.2,
 'good_recommendations': 3,
 'author_diversity': 0.8,
 'avg_similarity': 0.32448}
```
Berdasarkan hasil evaluasi, buku **The House in the Cerulean Sea** karya **T.J. Klune (rating 4.4)** sebagai referensi untuk menganalisis kualitas rekomendasi yang dihasilkan. Sistem menggunakan kombinasi fitur penulis dan rating yang diproses melalui metode TF-IDF dan cosine similarity untuk mengidentifikasi lima buku yang paling mirip. Hasil evaluasi menunjukkan bahwa dua rekomendasi teratas berasal dari penulis yang sama dengan nilai cosine similarity yang sangat tinggi (0.8187 dan 0.8037). Tiga rekomendasi lainnya berasal dari penulis berbeda namun tetap memiliki rating tinggi. Hal ini mengindikasikan bahwa sistem sangat efektif dalam mengidentifikasi kesamaan berdasarkan penulis, meskipun memiliki keterbatasan dalam mengenali aspek semantik lainnya seperti genre atau tema cerita karena keterbatasan fitur yang digunakan.
Secara keseluruhan, kualitas rekomendasi dapat dinilai cukup baik dengan rata-rata rating sebesar 4.20, yang lebih tinggi dari rata-rata dataset. Diversitas penulis juga menunjukkan hasil positif dengan 4 dari 5 rekomendasi berasal dari penulis berbeda, menandakan bahwa sistem tidak sepenuhnya bias terhadap satu penulis tertentu.
**Temuan Utama Evaluasi Content-Based**
- **Efektivitas Identifikasi Penulis**: Rekomendasi dari penulis yang sama mencapai cosine similarity di atas 0.8, menunjukkan sistem mampu mengenali keterkaitan kuat berdasarkan kesamaan penulis. Namun, tiga rekomendasi lainnya memiliki similarity 0.000 karena perbedaan penulis, sehingga menurunkan rata-rata cosine similarity menjadi 0.3245. Hal ini menunjukkan bahwa sistem sangat bergantung pada fitur penulis dalam menentukan kemiripan.
- **Kualitas Rating Rekomendasi**: Rata-rata rating rekomendasi mencapai 4.20 dengan 3 dari 5 rekomendasi memiliki rating ≥ 4.0, mengindikasikan bahwa sistem berhasil merekomendasikan buku-buku berkualitas tinggi. Hasil ini menunjukkan bahwa kombinasi fitur penulis dan rating efektif dalam menyaring konten berkualitas. Namun, pendekatan ini dapat mengabaikan buku-buku berkualitas dari penulis kurang terkenal.
- **Diversitas dan Keterbatasan Fitur**: Author diversity mencapai 0.80 menunjukkan variasi yang baik dalam rekomendasi, namun sistem masih terbatas karena hanya menggunakan fitur penulis dan rating. Keterbatasan ini menyebabkan sistem tidak dapat mengenali kesamaan yang lebih mendalam seperti tema, genre, atau gaya penulisan. Penambahan fitur konten yang lebih kaya diperlukan untuk meningkatkan akurasi rekomendasi.

### Evaluation Collaborative Filtering
* Metrik Evaluasi yang Digunakan: Model Collaborative Filtering ini bertujuan memprediksi rating yang mungkin diberikan pengguna pada buku. Oleh karena itu, metrik evaluasi kuantitatif standar untuk tugas regresi digunakan, yaitu Root Mean Squared Error (RMSE) dan Mean Absolute Error (MAE), dihitung pada validation set.
* Kesesuaian Metrik: RMSE dan MAE sangat sesuai karena mengukur secara langsung akurasi prediksi rating model terhadap rating sebenarnya yang diberikan pengguna. Ini sejalan dengan problem statement (memprediksi preferensi pengguna) dan solusi (model faktorisasi matriks berbasis NN).
* Penjelasan Metrik (Formula dan Cara Kerja):
   * RMSE (Root Mean Squared Error):
      * Formula: sqrt(mean((y_true - y_pred)^2))
      * Cara Kerja: Mengukur akar kuadrat dari rata-rata kuadrat perbedaan antara nilai aktual (y_true) dan nilai prediksi (y_pred). Dengan mengkuadratkan error, RMSE memberikan bobot lebih besar pada error yang besar (prediksi yang sangat salah). Hasilnya berada dalam satuan yang sama dengan variabel target (rating).
   * MAE (Mean Absolute Error):
      * Formula: mean(abs(y_true - y_pred))
      * Cara Kerja: Mengukur rata-rata dari nilai absolut perbedaan antara nilai aktual dan nilai prediksi. MAE memberikan gambaran langsung tentang rata-rata seberapa jauh prediksi model dari nilai sebenarnya, tanpa memberikan bobot ekstra pada error besar. Hasilnya juga dalam satuan yang sama dengan variabel target.
        
![image](https://github.com/user-attachments/assets/a8d65141-4beb-4362-9d34-159ea10a41bd)

```bash
==================================================
REKOMENDASI BUKU
==================================================

1. USER 65674
------------------------------
User ini pernah rating 7 buku
Buku favorit user:
  - Camp Jupiter Classified: A Probatios Jo (Rating: 5)
  - Every Missing Piece (Rating: 4)
  - Eden Mine (Rating: 3)

Rekomendasi buku:
  1. Lessons in Chemistry (Score: 3.03)
  2. Viviana Valentine Gets Her Man (Girl Fri (Score: 3.03)
  3. Birds: Photographs (Score: 3.03)
  4. Difficult Women: A History of Feminism i (Score: 3.03)
  5. My Dark Vanessa (Score: 3.02)
```
Berdasarkan hasil evaluasi menggunakan pendekatan neural matrix factorization untuk mempelajari hubungan antara pengguna dan buku berdasarkan pola interaksi historis mereka. Evaluasi dilakukan dengan memantau metrik pelatihan seperti loss, RMSE, dan MAE, serta menganalisis kualitas rekomendasi yang dihasilkan untuk pengguna tertentu.
Selama proses pelatihan, model menunjukkan pembelajaran yang efektif dengan penurunan training loss dari 0.998 ke 0.126 dan training RMSE dari 1.0 ke 0.355. Namun, terjadi peningkatan validation loss dan RMSE menjadi 1.218 setelah beberapa epoch, yang mengindikasikan terjadinya overfitting dimana model terlalu menyesuaikan dengan data pelatihan namun kurang mampu menggeneralisasi pada data baru.Dari segi rekomendasi, model menghasilkan skor prediksi yang relatif stabil dalam kisaran 3.02-3.04 untuk semua pengguna, menunjukkan kecenderungan konservatif dalam prediksi. Hal ini dapat disebabkan oleh strategi normalisasi atau regularisasi model, namun juga mengindikasikan kurangnya personalisasi yang kuat dalam rekomendasi yang dihasilkan.
**Temuan Utama Evaluasi Collaborative Filtering**:
- **Performa Learning dan Overfitting**: Training RMSE berhasil turun dari 1.0 ke 0.355 menunjukkan bahwa model mampu belajar pola dari data pelatihan dengan baik. Namun, validation RMSE justru meningkat dari 1.0 ke 1.218, mengindikasikan terjadinya overfitting yang signifikan. Fenomena ini menunjukkan perlunya penerapan teknik regularisasi yang lebih baik untuk meningkatkan kemampuan generalisasi model.
- **Konservatisme Prediksi**: Model menghasilkan prediksi rating dalam rentang sempit 3.02-3.04 untuk semua pengguna, menunjukkan kecenderungan prediksi yang terlalu konservatif. Hal ini mengindikasikan bahwa model belum mampu menangkap variasi preferensi pengguna secara optimal. Strategi normalisasi atau regularisasi yang diterapkan mungkin terlalu restriktif sehingga mengurangi sensitivitas model terhadap perbedaan preferensi.
- **Personalisasi dan Cold Start**: Model memberikan rekomendasi yang stabil namun kurang menunjukkan personalisasi yang kuat, terlihat dari pengguna dengan rating tinggi (5) hanya mendapat prediksi sekitar 3.0. Tantangan cold start tetap menjadi masalah utama karena model memerlukan data interaksi yang cukup untuk berfungsi optimal. Diperlukan pendekatan khusus untuk menangani pengguna baru atau yang memiliki sedikit riwayat interaksi.

## Kesimpulan
Berdasarkan evaluasi komprehensif terhadap kedua model, terlihat bahwa Content-Based Filtering dan Collaborative Filtering memiliki karakteristik yang saling melengkapi dalam konteks sistem rekomendasi buku.
Content-Based Filtering menunjukkan keunggulan dalam situasi dengan informasi buku yang jelas namun data pengguna yang terbatas. Model ini sangat efektif dalam memanfaatkan fitur eksplisit seperti penulis dan rating, menjadikannya ideal untuk rekomendasi cepat dan langsung. Namun, pendekatan ini memiliki keterbatasan signifikan dalam personalisasi karena tidak memanfaatkan pola perilaku pengguna lain, sehingga tidak dapat menawarkan rekomendasi yang bersifat eksploratif atau di luar domain yang sudah familiar.
Collaborative Filtering, khususnya dengan neural matrix factorization, menunjukkan potensi kuat dalam menangkap pola tersembunyi dari perilaku kolektif pengguna. Model ini berpotensi memberikan rekomendasi yang lebih personal karena berbasis pada interaksi nyata pengguna. Namun, model ini sangat bergantung pada ketersediaan data interaksi yang kaya dan rentan terhadap overfitting tanpa kontrol yang tepat. Hasil evaluasi menunjukkan bahwa meskipun model belajar dengan baik pada data pelatihan, kemampuan generalisasinya menurun pada data validasi.

**Strategi Pengembangan Lanjutan**
- **Pendekatan Hybrid**: Implementasi sistem hybrid yang menggabungkan kekuatan kedua model akan memberikan solusi yang lebih robust dan akurat. Content-Based Filtering dapat mengatasi masalah cold start untuk pengguna baru, sementara Collaborative Filtering dapat diaktifkan setelah pengguna memiliki riwayat interaksi yang memadai. Pendekatan ini akan memaksimalkan keunggulan masing-masing metode sambil meminimalkan kelemahan inherennya.
- **Pengembangan Fitur Konten**: Penambahan fitur konten yang lebih kaya seperti genre, sinopsis, tahun terbit, dan metadata lainnya akan meningkatkan kemampuan Content-Based Filtering dalam mengenali kesamaan semantik. Hal ini akan mengurangi ketergantungan berlebihan pada fitur penulis dan rating. Pengembangan fitur ini juga dapat diintegrasikan ke dalam model hybrid untuk memberikan konteks yang lebih kaya.
- **Optimalisasi Model Collaborative**: Penerapan teknik regularisasi dan validasi silang akan membantu mengatasi masalah overfitting pada model Collaborative Filtering. Selain itu, implementasi threshold-based recommendation dapat mengurangi kecenderungan prediksi yang terlalu konservatif. Teknik ensemble learning juga dapat dipertimbangkan untuk meningkatkan stabilitas dan akurasi prediksi.


