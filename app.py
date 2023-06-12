import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import pickle

st.title("PENAMBANGAN DATA")

data_set_description, upload_data, preprocessing, pca, modeling, implementation, profile = st.tabs(["Data Set Description", "Upload Data", "Preprocessing", "PCA", "Modeling", "Implementation", "Profile"])

with profile:
    st.write("Nama  : Nabila Atira Qurratul Aini ")
    st.write("Nim   : 210411100066 ")
    st.write("Kelas : Penambangan Data B ")
    st.write("Email : nabilatiraqurratul@gmail.com")


with data_set_description:
    st.write("""### DATA SET DESCRIPTION""")
    st.write("Data Set Ini Adalah : Harumanis Mango Physical Measurements (Pengukuran Fisik Mangga Harumanis) Dataset ini berisi 67 data pengukuran fisik tabular Mangga Harumanis.")
    st.write("Mangga atau mempelam adalah nama sejenis buah, demikian pula nama pohonnya. Mangga termasuk ke dalam genus Mangifera, yang terdiri dari 35-40 anggota dari famili Anacardiaceae.mangga arumanis (Mangifera indica L.) karena rasanya manis dan harum (arum) baunya. Daging buah tebal, lunak berwarna kuning, dan tidak berserat (serat sedikit). Aroma harum, tak begitu berair. Rasanya manis, tapi bagian ujung kadang kadang masih ada rasa asam.")
    
    st.write("### SUMBER DATA SET KAGGLE : ")
    st.write("https://www.kaggle.com/datasets/mohdnazuan/harumanis-mango-physical-measurement")
    
    st.write("""### PENJELASAN SETIAP KOLOM : """)
    st.write("""1.Weight(Berat):Berat mangga dalam gram (g)""")
    st.write("""2.Length (Panjang):Panjang buah mangga dalam centimeter (cm). Sebuah mangga arumanis panjangnya sekitar 15 cm dengan berat per buah 450 gram (rata-rata).  """)
    st.write("""3.Circumference (Lingkar): Lingkar mangga dalam sentimeter (cm)""")
    
    st.write("### APLIKASI UNTUK :")
    st.write("Pengukuran Fisik Mangga Harumanis")
    
    st.write("### SOURCE CODE APLIKASI ADA DI GITHUB : ")
    st.write("https://github.com/NabilaAtiraQurratulAini/UAS-PendataB")

with upload_data:
    df = pd.read_csv('https://raw.githubusercontent.com/NabilaAtiraQurratulAini/UAS-PendataB/main/data-pendat.csv')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    df = df.drop(columns=["No"])
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['Grade'])
    y = df['Grade'].values
    df
    X
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.Grade).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '0' : [dumies[0]],
        '1' : [dumies[1]],
    })

    st.write(labels)

with pca:
    st.subheader("Principal Component Analysis (PCA)")
    
    # Apply PCA on scaled features
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)

    # Create a new DataFrame with PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=["Principal Component 1", "Principal Component 2"])
    pca_df["Grade"] = df["Grade"]

    # Visualize the PCA results
    st.write("PCA Results:")
    st.write(pca_df)

    # Plot the PCA results
    chart = alt.Chart(pca_df).mark_circle().encode(
        x="Principal Component 1",
        y="Principal Component 2",
        color="Grade"
    )
    st.altair_chart(chart, use_container_width=True)

    pca_nb = GaussianNB()
    pca_nb = pca_nb.fit(pca_result, df["Grade"])
    pca_nb_pred = pca_nb.predict(pca_result)
    pca_nb_accuracy = round(100 * accuracy_score(df["Grade"], pca_nb_pred))

    # K-Nearest Neighbors with PCA
    pca_knn = KNeighborsClassifier(n_neighbors=10)
    pca_knn.fit(pca_result, df["Grade"])
    pca_knn_pred = pca_knn.predict(pca_result)
    pca_knn_accuracy = round(100 * accuracy_score(df["Grade"], pca_knn_pred))

    # Decision Tree with PCA
    pca_dt = DecisionTreeClassifier()
    pca_dt.fit(pca_result, df["Grade"])
    pca_dt_pred = pca_dt.predict(pca_result)
    pca_dt_accuracy = round(100 * accuracy_score(df["Grade"], pca_dt_pred))

    # Artificial Neural Network (Backpropagation) with PCA
    pca_ann_model = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=200)
    pca_ann_model.fit(pca_result, df["Grade"])
    pca_ann_pred = pca_ann_model.predict(pca_result)
    pca_ann_accuracy = round(100 * accuracy_score(df["Grade"], pca_ann_pred))

    # Save PCA models with pickle
    with open('pca_nb_model.pkl', 'wb') as file:
        pickle.dump(pca_nb, file)
    with open('pca_knn_model.pkl', 'wb') as file:
        pickle.dump(pca_knn, file)
    with open('pca_dt_model.pkl', 'wb') as file:
        pickle.dump(pca_dt, file)
    with open('pca_ann_model.pkl', 'wb') as file:
        pickle.dump(pca_ann_model, file)

    st.subheader("PCA Model Accuracies")
    data_pca = pd.DataFrame({
        "Model": ["Gaussian Naive Bayes", "K-Nearest Neighbors", "Decision Tree", "Artificial Neural Network (Backpropagation)"],
        "Accuracy": [pca_nb_accuracy, pca_knn_accuracy, pca_dt_accuracy, pca_ann_accuracy]
    })
    st.write(data_pca)

with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        ann = st.checkbox('Artificial Neural Network (Backpropagation)')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        #ANNBP
        ann_model = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=200)
        ann_model.fit(training, training_label)
        ann_pred = ann_model.predict(test)
        ann_akurasi = round(100 * accuracy_score(test_label, ann_pred))

        # Save models with pickle
        if naive:
            with open('gaussian_model.pkl', 'wb') as file:
                pickle.dump(gaussian, file)
        if k_nn:
            with open('knn_model.pkl', 'wb') as file:
                pickle.dump(knn, file)
        if destree:
            with open('dt_model.pkl', 'wb') as file:
                pickle.dump(dt, file)
        if ann:
            with open('ann_model.pkl', 'wb') as file:
                pickle.dump(ann_model, file)

        if submitted:
            if naive:
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn:
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree:
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
            if ann:
                st.write("Model Artificial Neural Network (Backpropagation) accuracy score : {0:0.2f}".format(ann_akurasi))
            
            # Save accuracies
            if naive:
                with open('gaussian_accuracy.pkl', 'wb') as file:
                    pickle.dump(gaussian_akurasi, file)
            if k_nn:
                with open('knn_accuracy.pkl', 'wb') as file:
                    pickle.dump(knn_akurasi, file)
            if destree:
                with open('dt_accuracy.pkl', 'wb') as file:
                    pickle.dump(dt_akurasi, file)
            if ann:
                with open('ann_accuracy.pkl', 'wb') as file:
                    pickle.dump(ann_akurasi, file)

        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi, ann_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree', 'Artificial Neural Network (Backpropagation)'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
# Tambahkan implementasi di sini
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        Weight = st.number_input('Masukkan weight (berat) : ')
        Lenght = st.number_input('Masukkan lenght (panjang) : ')
        Circumference = st.number_input('Masukkan Circumference(keliling) : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                            ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree', 'Artificial Neural Network (Backpropagation)'))
        use_pca = st.checkbox("Gunakan PCA")

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                Weight,
                Lenght,
                Circumference
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if use_pca:
                input_pca = pca.transform(input_norm)
                input_pred = None

                if model == 'Gaussian Naive Bayes':
                    with open('pca_nb_model.pkl', 'rb') as file:
                        pca_nb = pickle.load(file)
                    input_pred = pca_nb.predict(input_pca)
                elif model == 'K-NN':
                    with open('pca_knn_model.pkl', 'rb') as file:
                        pca_knn = pickle.load(file)
                    input_pred = pca_knn.predict(input_pca)
                elif model == 'Decision Tree':
                    with open('pca_dt_model.pkl', 'rb') as file:
                        pca_dt = pickle.load(file)
                    input_pred = pca_dt.predict(input_pca)
                elif model == 'Artificial Neural Network (Backpropagation)':
                    with open('pca_ann_model.pkl', 'rb') as file:
                        pca_ann_model = pickle.load(file)
                    input_pred = pca_ann_model.predict(input_pca)

                st.subheader('Hasil Prediksi')
                st.write('Menggunakan PCA dan Pemodelan:', model)
                st.write(input_pred)

            else:
                mod = None

                if model == 'Gaussian Naive Bayes':
                    with open('gaussian_model.pkl', 'rb') as file:
                        mod = pickle.load(file)
                elif model == 'K-NN':
                    with open('knn_model.pkl', 'rb') as file:
                        mod = pickle.load(file)
                elif model == 'Decision Tree':
                    with open('dt_model.pkl', 'rb') as file:
                        mod = pickle.load(file)
                elif model == 'Artificial Neural Network (Backpropagation)':
                    with open('ann_model.pkl', 'rb') as file:
                        mod = pickle.load(file)

                input_pred = mod.predict(input_norm)

                st.subheader('Hasil Prediksi')
                st.write('Menggunakan Pemodelan:', model)
                st.write(input_pred)