import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.set_option('deprecation.showPyplotGlobalUse', False)

# Sidebar options
menu = st.sidebar.radio('Pilih Menu', ['Data', 'K-Means Clustering'])

debug = st.sidebar.checkbox('Tampilkan Iterasi', value=False)
ntn = st.sidebar.checkbox('Pindahkan Centroid dengan nilai rata-rata NaN ke (0,0)', value=True)

def jarak(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def lakukan_kmeans_clustering(k, X, random_state):
    global debug

    np.random.seed(random_state)

    initial_centroids_indices = np.random.choice(len(X), k, replace=False)
    centroids = X[initial_centroids_indices]

    centers = centroids

    st.markdown('Centroid Awal:')
    st.write(centers)

    labels = np.random.choice(range(k), len(X))  
    jarak_poin = np.zeros((len(X), k))

    langkah = 1

    while True:
        if debug:
            expander = st.expander(f'Iterasi {langkah}', expanded=False)

        old_centers = centers

        for i in range(k):
            for j in range(len(X)):
                jarak_poin[j, i] = jarak(X[j], centers[i])

        labels = np.argmin(jarak_poin, axis=1)

        centers = np.array([X[labels == i].mean(0) for i in range(k)])

        if ntn:
            centers = np.nan_to_num(centers)

        if debug:
            with expander:
                df_debug = pd.DataFrame(X, columns=['price', 'sales', 'retail_price'])
                for i in range(k):
                    df_debug[f'Jarak ke Cluster {i+1}'] = jarak_poin[:, i]
                df_debug['Cluster Terdekat'] = labels + 1
                sns.scatterplot(x='price', y='sales', data=df_debug, hue=labels, palette='rainbow', legend='full')
                plt.scatter(old_centers[:, 0], old_centers[:, 1], color='black', s=100)
                plt.title(f'Iterasi {langkah}')
                st.pyplot()
                st.markdown('Jarak ke Setiap Cluster:')
                st.dataframe(df_debug.iloc[:, 3:], width=1000, height=500)
                st.markdown('Centroid Sebelumnya:')
                st.write(old_centers)
                st.markdown('Centroid Baru:')
                st.write(centers)
                st.markdown('---')

        if np.array_equal(centers, old_centers):
            unique, counts = np.unique(labels, return_counts=True)
            st.markdown('## Hasil K-Means Clustering')
            cluster_summary = pd.DataFrame({'Cluster': unique + 1, 'Jumlah Data': counts})
            st.write(cluster_summary)
            break

        langkah += 1

        if langkah > 30:
            st.markdown('Batas 30 langkah ditambahkan untuk menghindari loop tak terbatas.')
            break

    return centers, labels

def plot(X, centroids, labels, random_state):
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x='price', y='sales', data=X, hue=labels, palette='rainbow', legend='full')
    
    # Annotate centroids with numbers
    for i, centroid in enumerate(centroids):
        plt.text(centroid[0], centroid[1], str(i+1), fontsize=20, fontweight='bold', ha='center', va='center')
    
    plt.title(f'Final, dengan {len(centroids)} cluster dan RandomState {random_state} dan {len(df)} titik.')
    plt.legend(title='Clusters')
    st.pyplot()

# Data
data = {
    'title': ['Converse Chuck Taylor All-Star 70 Hi Black', 'Nike Air Monarch IV White Navy',
              'Nike Air Force 1 Low 07 White', 'adidas originals Superstar Cloud White Core Black Gold (GS)',
              'Nike Dunk Low Retro White Black Panda', 'Vans Old Skool Black White',
              'Nike Dunk Low Retro White Black Panda (W)', 'Nike Dunk Low Retro White Black Panda (GS)',
              'Nike M2K Tekno Cool White (Women\'s)', 'Nike Air Force 1 Low \'07 White (Women\'s)',
              'Nike Air Force 1 Low Flax', 'Vans SK8 HI Skateboarding Shoes Unisex',
              'New Balance NB 530 White Silver Navy', 'adidas originals Superstar Core Black Cloud White',
              'adidas originals Ozweego Core Black', 'Nike Flight Legacy Red White', 'New Balance 327 Grey',
              'Converse Chuck Taylor All-Star 70 Ox Black White', 'adidas originals Samba OG Cloud White Core Black',
              'Jordan 1 Low Wolf Grey (Women\'s)', 'Jordan 1 Mid \'Light Smoke Grey\'',
              'Jordan Legacy 312 Low Tech Grey Fire Red Black', 'Nike Air Force 1 Low \'07 White Black Pebbled Leather',
              'Nike Air Max 97 Triple White Wolf Grey (2017/2023)', 'Jordan 1 Mid White Metallic Gold Obsidian',
              'Nike Hyperdunk X Low White Pure Platinum', 'UGG Classic Ultra Mini Boot Chestnut (Women\'s)',
              'Jordan 1 Mid Wolf Grey Aluminum (Women\'s)', 'adidas originals Yeezy Boost 350 V2 Bone',
              'Air Jordan 11 Retro \'Bred\' (2019)', 'adidas originals Ozweego Trace Cargo'],
    'price': [82, 81, 105, 73, 90, 73, 95, 75, 99, 105, 122, 105, 105, 112, 55, 71, 89, 74, 86, 110, 116, 123, 113, 123, 135, 96, 86, 106, 216, 287, 81],
    'sales': [10, 4, 81, 3, 278, 4, 142, 47, 19, 19, 12, 2, 270, 1, 1, 19, 16, 0, 121, 219, 80, 19, 9, 16, 11, 4, 172, 30, 34, 112, 1],
    'retail_price': [90, 80, 115, 80, 115, 70, 115, 90, 120, 115, 135, 75, 95, 100, 120, 90, 100, 80, 100, 100, 125, 130, 115, 175, 110, 120, 150, 115, 230, 220, 110]
}

df = pd.DataFrame(data)

# Normalisasi Data
normalisasi = st.radio('Metode Normalisasi', ['None', 'MinMaxScaler'])

if normalisasi == 'MinMaxScaler':
    scaler = MinMaxScaler()
    df[['price', 'sales', 'retail_price']] = scaler.fit_transform(df[['price', 'sales', 'retail_price']])

# K-Means Clustering
if menu == 'K-Means Clustering':
    st.markdown('# Perhitungan K-Means Clustering')
    
    k = st.sidebar.number_input('k', min_value=1, max_value=100, value=3)
    random_state = st.sidebar.number_input('Random State', min_value=1, max_value=10000, value=30)
    
    st.write(f"Parameter yang digunakan:\n- k = {k}\n- random_state = {random_state}")

    X = df[['price', 'sales', 'retail_price']].values

    centroids, labels = lakukan_kmeans_clustering(k, X, random_state)
    plot(df, centroids, labels, random_state)

# Display Data
elif menu == 'Data':
    st.markdown('# Data yang Digunakan')
    st.dataframe(df[['title', 'price', 'sales', 'retail_price']], width=1000, height=500)
