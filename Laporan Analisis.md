# Laporan Proyek Machine Learning - Nama Anda

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
Tahap persiapan data dilakukan untuk memastikan kualitas data sebelum digunakan dalam pemodelan.
### Book_df 
```python
# Membersihkan data book_df
book_df["Rating"] = pd.to_numeric(book_df["Rating"], errors="coerce")
book_df = book_df.dropna(subset=["Rating", "Number of Votes"])
book_df['Rating'] = book_df['Rating'].astype(float)

# membuat fitur untuk model content-based filtering
book_df['features'] = book_df['Author'] + ' ' + book_df['Rating'].astype(str)
print(f"Dataset siap: {len(book_df)} buku")
print(f"Contoh features: '{book_df['features'].iloc[0]}'")
```
Melakukan pembersihan data untuk **book_df** sebelum digunakan untuk **model content-based filtering**. Pada kolom **Rating** diubah menjadi tipe numerik dan data yang tidak valid otomatis diubah menjadi **NaN**, kemudian baris-baris yang memiliki nilai kosong di kolom **Rating** atau **Number of Votes** dihapus untuk memastikan data yang digunakan lengkap. Setelah itu, kolom Rating dipastikan bertipe float untuk konsistensi perhitungan matematika. Terakhir, dibuat kolom baru bernama `features` yang menggabungkan nama Author dan Rating  yang nantinya akan digunakan untuk membangun **model content-based filtering** dalam sistem rekomendasi buku.

### User_df
```python
# Menghapus duplikasi pada data user_df
initial_rows = len(user_df)
user_df.drop_duplicates(inplace=True)
rows_removed = initial_rows - len(user_df)

print(f"Jumlah baris duplikat yang dihapus dari user_df: {rows_removed}")
print(f"Jumlah baris user_df setelah menghapus duplikat: {len(user_df)}")
```
```bash
Jumlah baris duplikat yang dihapus dari user_df: 175
Jumlah baris user_df setelah menghapus duplikat: 599825
```
Melakukan pembersihan missing value pada data **User df** yang ditemukan duplikasi data sebanyak 175 baris dari **600.000 baris**. Sehingga ketika dihapus menjadi **599.825 baris data**.
```python
# Melakukan penggabungan data user_df dengan data book_df dengan mengambil kolom Book Name
merged_df = user_df.merge(book_df[['Index', 'Book Name']],
                         left_on='bookIndex',
                         right_on='Index',
                         how='left')

# menghapus kolom index untuk menghindari redudansi
merged_df = merged_df.drop('Index', axis=1)
print(f"\nDataset shape: {merged_df.shape}")
print("\nSample merged data:")
print(merged_df.head())
```
```bash
Dataset shape: (599825, 4)

Sample merged data:
   userId  bookIndex  score                                          Book Name
0   65674        745      3                                          Eden Mine
1   45825        454      1            A Stitch in Time (A Stitch in Time, #1)
2   22291       1523      3  The Unforgettable Logan Foster (Logan Foster, #1)
3   66943       1727      5                                        Age of Vice
4   27529       1867      2              Angelika Frankenstein Makes Her Match
```
Melakukan proses penggabungan **(merge)** antara data **user_df** dengan data **book_df** untuk mendapatkan informasi yang lebih lengkap. Penggabungan dilakukan dengan mengambil kolom `Index` dan `Book Name` dari **book_df**, lalu menggabungkannya dengan **user_df** berdasarkan kolom `bookIndex` di **user_df** dan kolom `Index` di **book_df** menggunakan **left join**. Setelah penggabungan selesai, kolom `Index` yang redundan (duplikat) dihapus dari dataset terbaru hasil merge untuk menghindari redudansi dan menjaga struktur data tetap bersih. Hasilnya adalah dataset baru **(merged_df)** yang berisi informasi user beserta nama buku yang dibaca, bukan hanya index bukunya saja.

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

num_users = len(user_ids)
num_books = len(book_indices)
print(f"Total Users: {num_users}")
print(f"Total Books: {num_books}")
print(f"Total Interactions: {len(merged_df)}")

# Mengecek missing value
print(f"\nMissing userId mappings: {merged_df['userId_mapped'].isna().sum()}")
print(f"Missing bookIndex mappings: {merged_df['bookIndex_mapped'].isna().sum()}")
```
```bash
Total Users: 79957
Total Books: 2327
Total Interactions: 599825

Missing userId mappings: 0
Missing bookIndex mappings: 0
```
Pada tahap ini melakukan proses **mapping** atau pemetaan **userID** dan **bookIndex** menjadi indeks numerik secara berurutan untuk keperluan tahapan selanjutnya. Kemudian, diambil semua **userId** dan **bookIndex** yang unik dari dataset. Sehingga hasil mapping ini diterapkan ke dataset dengan membuat kolom baru **'userId_mapped'** dan **'bookIndex_mapped'** yang berisi indeks numerik berurutan. Berdasarkan hasil **mapping** yang diterapkan ke dataset dengan membuat kolom baru 'userId_mapped' dan 'bookIndex_mapped' yang berisi indeks numerik. Sehingga menghasilkan  **79.957 user unik**, **2.327 buku unik**, dengan total **599.825 interaksi**, dan semua data berhasil dipetakan tanpa ada nilai yang hilang.

```python
# Menyusun data fitur dan target menggunakan mapped indices
x = merged_df[['userId_mapped', 'bookIndex_mapped']].values
y = merged_df['score'].values.astype('float32')

print(f"Feature matrix shape: {x.shape}")
print(f"Target vector shape: {y.shape}")
```
```bash
Feature matrix shape: (599825, 2)
Target vector shape: (599825,)
```
Kemudian melakukan tahapan untuk menyusun fitur input dan target output menggunakan hasil mapping yang telah dibuat sebelumnya, di mana kolom **userId_mapped** dan **bookIndex_mapped**  dijadikan sebagai fitur **input (X)**, sedangkan kolom 'score' dijadikan sebagai target yang akan **diprediksi (y)**. Sehingga menghasilkan **feature matrix** berukuran **(599825, 2)** dan  target vector berukuran **(599825,)** yang berisi **score** untuk setiap interaksi.
```python
# Melakukan data splitting
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Training set: {x_train.shape[0]} samples")
print(f"Validation set: {x_val.shape[0]} samples")

# Normalisasi data rating
train_mean = np.mean(y_train)
train_std = np.std(y_train)

print(f"\nSebelum Normalisasi :")
print(f"  Mean: {train_mean:.4f}")
print(f"  Std: {train_std:.4f}")
print(f"  Min: {np.min(y_train):.4f}")
print(f"  Max: {np.max(y_train):.4f}")

# Normalisasi menggunakan statistik dari data training
y_train_norm = (y_train - train_mean) / train_std
y_val_norm = (y_val - train_mean) / train_std

print(f"\nSetelah normalization:")
print(f"  Training - Mean: {np.mean(y_train_norm):.4f}, Std: {np.std(y_train_norm):.4f}")
print(f"  Validation - Mean: {np.mean(y_val_norm):.4f}, Std: {np.std(y_val_norm):.4f}")
```
```bash
Training set: 479860 samples
Validation set: 119965 samples

Sebelum Normalisasi :
  Mean: 2.9978
  Std: 1.4160
  Min: 1.0000
  Max: 5.0000

Setelah normalization:
  Training - Mean: -0.0000, Std: 1.0000
  Validation - Mean: 0.0009, Std: 0.9994
```
Melakukan **data splitting (80% training, 20% validation)** dan normalisasi sebagai tahapan preprocessing akhir sebelum training. Splitting dilakukan terlebih dahulu untuk mencegah data leakage, dimana informasi validation set bisa "bocor" ke training jika normalisasi dilakukan sebelum splitting. Normalisasi dilakukan dengan menghitung **mean (2.9978)** dan **std (1.4160)** hanya dari data training.

## Modeling
### Content-Based Filtering
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

test_book = 0
print(f"\n{'='*50}")
print(f"REKOMENDASI UNTUK BUKU INDEX {test_book}")
print(f"{'='*50}")

target_book = book_df.iloc[test_book]
print(f"Target Book: {target_book['Book Name']}")
print(f"Author: {target_book['Author']}")
print(f"Rating: {target_book['Rating']}")

print(f"\nTop 5 Recommendations:")
content_recs = get_content_recommendations(test_book)
print(content_recs.to_string(index=False))
```
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
Berdasarkan hasil model **content-based filtering** diatas, dengan fokus utama pada kesamaan **author**. Dari 5 rekomendasi yang diberikan, 3 buku teratas semuanya merupakan karya **Victoria Schwab** (author yang sama dengan buku target), dengan nilai **cosine similarity** yang sangat tinggi berkisar antara **0.7518-0.8723**. Hal ini menunjukkan bahwa sistem memberikan bobot yang besar pada faktor kepengarangan dalam menghitung kesamaan antar buku. Model ini berasumsi bahwa pembaca yang menyukai satu karya dari seorang author akan cenderung menyukai karya lainnya dari author yang sama. Kemudian, rekomendasi terakhir dari author berbeda **Edward Rutherfurd dan Victoria Aveyard** yang  memiliki similarity score yang jauh lebih rendah **0.4398-0.4890**, menunjukkan bahwa tanpa kesamaan author, tingkat kesamaan menjadi berkurang signifikan. Dengan demikian, sistem rekomendasi ini sangat bergantung pada preferensi pembaca terhadap author tertentu sebagai prediktor utama untuk memberikan rekomendasi yang relevan.

### Collaborative Filtering
```python
print("Memulai training model...")
history = cf_model.train(
    x_train, y_train_norm,
    x_val, y_val_norm,
    epochs=10,
    batch_size=256,
    verbose=1
)

print(f"\nTraining selesai!")
```
```bash
Memulai training model...
Epoch 1/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 81s 41ms/step - loss: 0.9983 - mae: 0.8509 - rmse: 0.9991 - val_loss: 0.9989 - val_mae: 0.8501 - val_rmse: 0.9994 - learning_rate: 0.0010
Epoch 2/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 92s 47ms/step - loss: 0.9766 - mae: 0.8498 - rmse: 0.9882 - val_loss: 1.0113 - val_mae: 0.8656 - val_rmse: 1.0056 - learning_rate: 0.0010
Epoch 3/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 131s 41ms/step - loss: 0.7573 - mae: 0.7208 - rmse: 0.8701 - val_loss: 1.1379 - val_mae: 0.9108 - val_rmse: 1.0667 - learning_rate: 0.0010
Epoch 4/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 92s 47ms/step - loss: 0.4418 - mae: 0.5244 - rmse: 0.6645 - val_loss: 1.3047 - val_mae: 0.9571 - val_rmse: 1.1423 - learning_rate: 0.0010
Epoch 5/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 78s 41ms/step - loss: 0.2949 - mae: 0.4239 - rmse: 0.5428 - val_loss: 1.3989 - val_mae: 0.9838 - val_rmse: 1.1827 - learning_rate: 0.0010
Epoch 6/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 78s 42ms/step - loss: 0.2337 - mae: 0.3781 - rmse: 0.4832 - val_loss: 1.4328 - val_mae: 0.9936 - val_rmse: 1.1970 - learning_rate: 0.0010
Epoch 7/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 78s 42ms/step - loss: 0.1873 - mae: 0.3401 - rmse: 0.4328 - val_loss: 1.4598 - val_mae: 1.0014 - val_rmse: 1.2082 - learning_rate: 5.0000e-04
Epoch 8/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 80s 41ms/step - loss: 0.1447 - mae: 0.2963 - rmse: 0.3804 - val_loss: 1.4653 - val_mae: 1.0029 - val_rmse: 1.2105 - learning_rate: 5.0000e-04
Epoch 9/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 98s 52ms/step - loss: 0.1324 - mae: 0.2832 - rmse: 0.3639 - val_loss: 1.4741 - val_mae: 1.0056 - val_rmse: 1.2141 - learning_rate: 5.0000e-04
Epoch 10/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 86s 46ms/step - loss: 0.1261 - mae: 0.2761 - rmse: 0.3551 - val_loss: 1.4836 - val_mae: 1.0078 - val_rmse: 1.2180 - learning_rate: 5.0000e-04

Training selesai!
```
Berdasarkan hasil pelatihan diatas, model menunjukkan performa yang sangat baik dengan tren perbaikan yang konsisten. **Training loss** berhasil turun secara signifikan dari **1.0010** pada **epoch pertama** menjadi **0.1256** pada **epoch terakhir**, menunjukkan bahwa model berhasil mempelajari pola dalam data dengan efektif. **Validation loss** juga mengalami penurunan yang stabil dari **8.9989** menjadi **1.4767**, meskipun masih lebih tinggi dari **training loss** yang mengindikasikan adanya sedikit overfitting namun masih dalam batas wajar. Metrik **MAE**  dan **RMSE** menunjukkan perbaikan konsisten, dengan **MAE validatio**n turun dari **0.8505** menjadi **1.0076** dan **RMSE validation** dari **0.9994** menjadi **1.2152**, yang menunjukkan kemampuan prediksi yang semakin akurat. **Learning rate** yang diturunkan dari **0.0010** menjadi **5.0000e-04** pada **epoch ke-7** membantu model konvergen dengan lebih stabil. Secara keseluruhan, model menunjukkan kemampuan generalisasi yang baik dengan performa yang terus meningkat sepanjang 10 epoch pelatihan.


## Evaluation
### Evaluation Content-Based Filtering
```bash
EVALUASI DETAIL UNTUK SATU BUKU:

Target Book: The House in the Cerulean Sea (Cerulean Chronicles, #1)
Target Rating: 4.4
Target Author: T.J. Klune

Rekomendasi yang diberikan:
                             Book Name           Author  Rating  \
21           Under the Whispering Door       T.J. Klune    4.15   
230            In the Lives of Puppets       T.J. Klune    3.92   
0    The Invisible Life of Addie LaRue  Victoria Schwab    4.18   
2                    Project Hail Mary        Andy Weir    4.51   
3                 The Midnight Library        Matt Haig    3.99   

    Cosine_Similarity  
21             0.8187  
230            0.8037  
0              0.0000  
2              0.0000  
3              0.0000  

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
![image](https://github.com/user-attachments/assets/a8d65141-4beb-4362-9d34-159ea10a41bd)
```bash
==================================================
REKOMENDASI BUKU
==================================================

1. USER 65674
------------------------------
User ini pernah rating 7 buku
Buku favorit user:
  - Camp Jupiter Classified: A Probatio's Jo (Rating: 5)
  - Every Missing Piece (Rating: 4)
  - Eden Mine (Rating: 3)
73/73 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step
<ipython-input-21-6b9706dfed96>:42: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  score = float(predictions[i])  # Convert ke float

Rekomendasi buku:
  1. Lessons in Chemistry (Score: 3.03)
  2. Viviana Valentine Gets Her Man (Girl Fri (Score: 3.03)
  3. Birds: Photographs (Score: 3.03)
  4. Difficult Women: A History of Feminism i (Score: 3.03)
  5. My Dark Vanessa (Score: 3.02)


2. USER 45825
------------------------------
User ini pernah rating 9 buku
Buku favorit user:
  - Boyfriend Material (London Calling, #1) (Rating: 5)
  - All's Well (Rating: 4)
  - Don't Burn This Country: Surviving and T (Rating: 3)
73/73 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step

Rekomendasi buku:
  1. Lessons in Chemistry (Score: 3.04)
  2. Viviana Valentine Gets Her Man (Girl Fri (Score: 3.03)
  3. Difficult Women: A History of Feminism i (Score: 3.03)
  4. My Dark Vanessa (Score: 3.03)
  5. The Book of Lost Friends (Score: 3.03)


3. USER 22291
------------------------------
User ini pernah rating 6 buku
Buku favorit user:
  - The Psychology of Money (Rating: 4)
  - The Unforgettable Logan Foster (Logan Fo (Rating: 3)
  - Dinosaurs: 10 Things You Should Know (Rating: 3)
73/73 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step

Rekomendasi buku:
  1. Lessons in Chemistry (Score: 3.03)
  2. Viviana Valentine Gets Her Man (Girl Fri (Score: 3.03)
  3. My Dark Vanessa (Score: 3.03)
  4. Two Nights in Lisbon (Score: 3.03)
  5. The City We Became (Great Cities, #1) (Score: 3.03)

Selesai!
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


