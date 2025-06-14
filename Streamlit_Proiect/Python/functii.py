import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import math as mh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import geopandas as gpd


# ==================== VIZUALIZARE & EXPLORARE DATE ====================
def explorare_date(data, variables):
    st.title("Proiect Partea 1")
    afiseaza_extreme_speranta_viata(data)
    afiseaza_filtru_speranta_viata(data)
    compara_tari_pe_variabila(data, variables)
    afiseaza_statistici_descriptive(data, variables)

def afiseaza_extreme_speranta_viata(data):
    st.header("Speranta de viata")

    if st.button("Afiseaza tara cu cea mai mare speranta de viata"):
        max_row = data.loc[data['life_expectancy_at_birth'].idxmax()]
        st.write(f"{max_row['country']}: {max_row['life_expectancy_at_birth']} ani")

    if st.button("Afiseaza tara cu cea mai scazuta speranta de viata"):
        min_row = data.loc[data['life_expectancy_at_birth'].idxmin()]
        st.write(f"{min_row['country']}: {min_row['life_expectancy_at_birth']} ani")


def afiseaza_filtru_speranta_viata(data):
    min_val, max_val = st.slider(
        "Afiseaza tarile cu speranta de viata in intervalul ales:",
        min_value=int(data['life_expectancy_at_birth'].min()),
        max_value=int(data['life_expectancy_at_birth'].max()),
        value=(int(data['life_expectancy_at_birth'].min()), int(data['life_expectancy_at_birth'].max())),
        step=1
    )

    filtered_data = data[
        (data['life_expectancy_at_birth'] >= min_val) &
        (data['life_expectancy_at_birth'] <= max_val)
        ]

    st.write(f"Țările cu speranța de viață între {min_val} și {max_val} ani sunt:")
    st.dataframe(filtered_data[['country', 'life_expectancy_at_birth']])

    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_data['life_expectancy_at_birth'])
    plt.title("Distribuția Speranței de Viață")
    st.pyplot(plt)


def compara_tari_pe_variabila(data, variables):
    st.header("Compara tari intre ele")
    selected_variable = st.selectbox("Alege variabila pentru comparație:", variables)
    countries = data["country"].unique()
    selected_countries = st.multiselect("Alege tarile:", countries)

    if selected_countries and selected_variable:
        filtered = data[data["country"].isin(selected_countries)][["country", selected_variable]]
        st.dataframe(filtered)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="country", y=selected_variable, data=filtered)
        plt.xticks(rotation=45)
        plt.title(f"Comparatie intre tari pentru {selected_variable}")
        st.pyplot(plt)


def afiseaza_statistici_descriptive(data, variables):
    st.header("Statistici descriptive")
    countries_list = ["All"] + list(data["country"].unique())
    selected_countries = st.multiselect("Alege tarile:", countries_list)
    selected_variables = st.multiselect("Alege variabilele:", variables)

    if selected_countries and selected_variables:
        if "All" in selected_countries:
            selected_countries = countries_list[1:]
        filtered_stats = data[data["country"].isin(selected_countries)][selected_variables]

        stats_df = pd.DataFrame({
            "Medie": filtered_stats.mean(),
            "Mediana": filtered_stats.median(),
            "Nr. de NaN-uri": filtered_stats.isna().sum(),
            "Deviatia standard": filtered_stats.std(),
            "Minim": filtered_stats.min(),
            "Maxim": filtered_stats.max(),
            "Sumă": filtered_stats.sum(),
            "Q1": filtered_stats.quantile(0.25),
            "Q3": filtered_stats.quantile(0.75)
        })
        st.dataframe(stats_df)


def vizualizare_date(data, variables):
    st.header("Afisarea datelor")
    st.write(f"{data.shape[0]} rânduri și {data.shape[1]} coloane")
    st.dataframe(data)

    st.header("Filtrarea datelor")
    st.subheader("Dupa numele tarii")
    country = st.text_input("Introdu o tara:")
    if country:
        filtered_data_country = data[data['country'].str.contains(country, case=False, na=False)]
        st.dataframe(filtered_data_country)
    else:
        st.dataframe(data)

    st.header("Sortarea datelor")
    selected_variable = st.selectbox("Alege variabila:", variables)
    sort_order = st.radio("Alege ordinea de sortare:", ["Crescator", "Descrescator"])
    ascending = sort_order == "Crescator"
    sorted_data = data.sort_values(by=selected_variable, ascending=ascending)
    st.dataframe(sorted_data)




# ================================  EDA ================================
def afiseaza_tipuri_si_nan(data):
    st.subheader("Tipuri de date și valori lipsă")
    info_df = data.dtypes.rename("Tip de date").to_frame()
    info_df["Valori Lipsă"] = data.isna().sum()
    st.dataframe(info_df)

def analiza_valori_lipsa(data):
    st.subheader("Analiza valorilor lipsă")
    missing_vals = data.isnull().sum()
    missing_percent = (missing_vals / len(data)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_vals,
        'Percentage (%)': missing_percent
    }).query("`Missing Values` > 0").sort_values(by="Percentage (%)", ascending=False)

    if not missing_df.empty:
        st.write("### Tabel cu valori lipsă per coloană:")
        st.dataframe(missing_df)

        # grafic
        plt.figure(figsize=(8, 4))
        missing_df["Percentage (%)"].plot(kind='barh', color='orange')
        plt.title("Procentul valorilor lipsă per coloană")
        plt.xlabel("Procent (%)")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)

        st.info("Exemplu: `self_perceived_health` are ~3% valori lipsă, iar `at_risk_of_poverty_rate` ~7%.")


def tratare_valori_lipsa(data):
    st.subheader("Tratarea valorilor lipsă")
    method = st.radio("Alege metoda de imputare:", ["mean", "median"], horizontal=True)

    numeric_columns = data.select_dtypes(include="number").columns
    categorical_columns = data.select_dtypes(exclude="number").columns

    if method == "mean":
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    elif method == "median":
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    for col in categorical_columns:
        if data[col].isna().sum() > 0:
            data[col] = data[col].fillna(data[col].mode()[0])

    st.write("### Date după tratarea valorilor lipsă:")
    st.dataframe(data)
    return data

def genereaza_histograme(data):
    st.subheader("Histograme variabile numerice")
    numerical_cols = data.select_dtypes(include="number").columns
    n_cols = 3
    n_rows = mh.ceil(len(numerical_cols) / n_cols)

    plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    for i, col in enumerate(numerical_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(data[col].dropna(), bins=29, color='skyblue', edgecolor='black')
        plt.title(f'Distribuția: {col}')
        plt.xlabel(col)
        plt.ylabel('Frecvență')
    plt.tight_layout()
    st.pyplot(plt)

def afiseaza_distributie_variabila(data):
    st.subheader("Distribuția unei variabile selectate")
    variabile = data.select_dtypes(include="number").columns.tolist()
    var_selectata = st.selectbox("Alege variabila pentru distribuție:", variabile, key="distributie_var")

    if var_selectata:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[var_selectata], kde=True, color="blue")
        plt.title(f"Distribuția variabilei: {var_selectata}")
        plt.tight_layout()
        st.pyplot(plt)

def analiza_corelatii(data):
    st.subheader("Analiza relației între două variabile")
    variabile = data.select_dtypes(include="number").columns.tolist()
    var1 = st.selectbox("Prima variabilă:", variabile)
    var2 = st.selectbox("A doua variabilă:", variabile)

    if var1 and var2:
        st.write(f"Scatter plot între {var1} și {var2}")
        plt.figure(figsize=(10, 6))
        sns.regplot(x=var1, y=var2, data=data, scatter_kws={'s': 10}, line_kws={'color': 'red'})
        plt.title(f"Relația dintre {var1} și {var2}")
        plt.tight_layout()
        st.pyplot(plt)

def plot_pairplot_numeric(df):
    st.header("Plot de densitate")
    numeric_cols = df.select_dtypes(include=['number']).columns
    g = sns.pairplot(df[numeric_cols], diag_kind='kde')
    plt.suptitle("Pairplot pentru variabilele numerice", y=1.02)
    st.pyplot(g.fig)
    plt.close()
    st.write("Distribuția variabilelor (diagonala principală)")
    st.write("Fiecare histogramă arată cum sunt distribuite valorile fiecărei variabile.")
    st.write("Unele variabile par să aibă o distribuție aproximativ normală, în timp ce altele prezintă o distribuție asimetrică sau outliers.")
    st.write("Lipsa unor relații liniare clare între majoritatea variabilelor.")
    st.write("Posibile relații inverse și outliers evidențiate în unele perechi de variabile.")

def plot_boxplot_cat_numeric(df):
    st.header("Distributie pe țări")
    st.write("Punctele îndepărtate sunt outlieri. Pot afecta distribuția.")
    cat_col = 'country'
    numerical_cols = df.select_dtypes(include=['number']).columns
    num_col = st.selectbox('Selectează variabila numerică', numerical_cols)
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=cat_col, y=num_col, palette='viridis', fliersize=0)
    sns.stripplot(data=df, x=cat_col, y=num_col, color='red', jitter=True, size=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

def find_outliers_and_plot(df):
    st.header("Outliers (IQR)")
    st.write("Determinăm outlieri folosind metoda IQR.")
    num_cols = df.select_dtypes(include=['number']).columns
    outliers_columns = []

    def find_outliers_iqr(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_df = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        return outliers_df, lower_bound, upper_bound

    for col in num_cols:
        outliers, lower_bound, upper_bound = find_outliers_iqr(df, col)
        if not outliers.empty:
            outliers_columns.append(col)
            st.write(f"**Coloana:** {col}")
            st.write(f"**Număr de outlieri: {len(outliers)}**")
            st.write(f"**Limite IQR:** {lower_bound:.2f} – {upper_bound:.2f}")
            st.dataframe(outliers)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)
    return outliers_columns

def apply_log_transform(df, outliers_columns):
    st.header("Transformare logaritmică (pentru outlieri)")
    df_transformed = df.copy()
    for col in outliers_columns:
        df_transformed[col + "_log"] = np.log1p(df_transformed[col])
    df_transformed.drop(columns=outliers_columns, inplace=True)
    st.write("Datele după log-transformare:")
    st.dataframe(df_transformed)
    return df_transformed

def show_correlation_matrix(df):
    st.header("Matricea de corelație")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matricea de corelație pentru variabilele numerice")
    plt.tight_layout()
    st.pyplot(plt)


def encoding_demo(df):
    st.header("Encodificare")
    st.write("""
        Encodificarea transformă variabilele **categorice** în valori **numerice** pentru a fi utilizate în modele statistice.
        Deoarece nu avem variabile non-numerice, vom aplica encodificarea în scop demonstrativ.
    """)

    cat_col = st.selectbox("Selectează coloana categorică de encodat", df.select_dtypes(include='object').columns, index=0)
    method = st.radio("Metodă de encodare", ['Label Encoding', 'One-Hot Encoding'])

    if method == 'Label Encoding':
        le = LabelEncoder()
        df_encoded = df.copy()
        df_encoded[cat_col + '_encoded'] = le.fit_transform(df[cat_col])
        st.subheader("Label Encoded")
        st.write(df_encoded[[cat_col, cat_col + '_encoded']])

    elif method == 'One-Hot Encoding':
        ohe = OneHotEncoder(sparse=False)
        encoded_data = ohe.fit_transform(df[[cat_col]])
        encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out([cat_col]))
        df_encoded = pd.concat([df[[cat_col]].reset_index(drop=True), encoded_df], axis=1)
        st.subheader("One-Hot Encoded")
        st.write(df_encoded)


def standardizare_date(df):
    st.header("Standardizare date")
    st.write("Tehnică de preprocesare a datelor care transformă variabilele astfel încât să aibă o distribuție cu media 0 și abaterea standard 1. ")
    st.write("Ajută la obținerea unor rezultate corecte și stabile, la uniformizarea datelor și la îmbunătățirea performanței modelelor")

    numeric_data = df.select_dtypes(include=['number'])
    st.write("Media inițială:", numeric_data.mean().round(2).to_dict())
    st.write("Deviatia standard inițiala:", numeric_data.std().round(2).to_dict())

    scaler = StandardScaler()
    data_std = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns)

    st.subheader("Datele după standardizare")
    st.dataframe(data_std)

    st.write("Media după standardizare:", data_std.mean().round(6).to_dict())
    st.write("Deviatia standard dupa standardizare:", data_std.std().round(2).to_dict())

    return data_std

def normalizare_date(data_std):
    st.header("Normalizare date")
    st.write("Tehnică de preprocesare a datelor utilizată pentru a transforma valorile variabilelor astfel încât să fie într-un interval specificat, de obicei între 0 și 1.")
    st.write("Scopul este să aducă variabilele la o scară similară.")

    minmax_scaler = MinMaxScaler()
    data_norm = pd.DataFrame(minmax_scaler.fit_transform(data_std), columns=data_std.columns)

    st.write("Datele după normalizare")
    st.dataframe(data_norm)

    st.write("Valorile minime dupa normalizare:", data_norm.min().round(2).to_dict())
    st.write("Valorile maxime dupa normalizare:", data_norm.max().round(2).to_dict())

    return data_norm

def regresie_liniara_multipla(data_norm):
    st.header("Regresia Liniară Multiplă")

    available_targets = [
        "life_expectancy_at_birth",
        "current_health_expenditure",
        "total_air_emissions",
        "self_perceived_health",
        "at_risk_of_poverty_rate",
        "mortality_rate_infant_log",
        "self_reported_unmet_needs_for_medical_examination_log",
        "total_unemployment_rate_log",
        "hipc_log"
    ]

    st.write("Selectează variabila țintă")
    target = st.selectbox("Alege variabila țintă", available_targets, key="linear_target")

    X = data_norm.drop([target], axis=1)
    y = data_norm[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # antrenarea modelului
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Rezultatele modelului de regresie liniară:")
    st.write("Mean Absolute Error (MAE):", mae)
    st.write("Mean Squared Error (MSE):", mse)
    st.write("R2 Score:", r2)
    st.write("Coeficientul de determinare (R2) măsoară cât de bine explică modelul variația din date: R² = 1 înseamnă predicții perfecte, iar un R2 de 0 înseamnă că modelul nu explică deloc variația datelor.")

    # afisarea ecuației modelului
    st.subheader("Ecuația modelului de regresie:")
    coef_df = pd.DataFrame({
        'Variabilă': X.columns,
        'Coeficient': lr_model.coef_
    })
    st.write("Intercept:", lr_model.intercept_)
    st.dataframe(coef_df)
    return target
def corectie_model_regresie(data_norm, target):
    st.header("Corectia modelului de regresie - Random Forest vs XGBoost:")

    X = data_norm.drop(columns=[target])
    y = data_norm[target]

    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)

    st.write("Random Forest:")
    st.write("  MAE:", rf_mae)
    st.write("  MSE:", rf_mse)
    st.write("  R2:", rf_r2)

    st.write("\nXGBoost:")
    st.write("  MAE:", xgb_mae)
    st.write("  MSE:", xgb_mse)
    st.write("  R2:", xgb_r2)

    st.write("MAE măsoară media diferențelor absolute între valorile prezise de model și valorile reale (observate), fiind media absolută a erorilor.MAE mic indică un model care face erori mici, pe medie, în predicțiile sale.")
    st.write("MSE este media pătratelor diferențelor între valorile reale și valorile prezise de model. În comparație cu MAE, MSE penalizează erorile mari mult mai sever, deoarece se folosesc pătratele diferențelor.MSE mic indică un model care face erori mici și precise.")

def clusterizare_kmeans(preprocessed_df, numeric_columns, original_df):
    st.title("Clusterizare K-Means")

    numeric_data = preprocessed_df[numeric_columns]

    # PCA pentru vizualizare
    pca = PCA(n_components=2)
    numeric_data_pca = pca.fit_transform(numeric_data)

    n_clusters = st.slider("Selectează numărul de clustere", min_value=2, max_value=8, value=3, step=1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(numeric_data)
    preprocessed_df['kmeans_cluster'] = kmeans.labels_

    # readaugam country
    if 'country' in original_df.columns:
        countries = original_df['country'].reset_index(drop=True)
    else:
        countries = pd.Series([f"Țară {i}" for i in range(len(preprocessed_df))])  # fallback

    st.write(f"\n### Clusterizarea pentru {n_clusters} clustere:")
    for cluster_id in range(n_clusters):
        cluster_mask = preprocessed_df['kmeans_cluster'] == cluster_id
        countries_in_cluster = countries[cluster_mask].tolist()
        st.write(f"Cluster {cluster_id}: {', '.join(countries_in_cluster)}")

    silhouette_avg = silhouette_score(numeric_data, kmeans.labels_)
    st.write(f"Scorul Silhouette pentru {n_clusters} clustere: {silhouette_avg:.3f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(numeric_data_pca[:, 0], numeric_data_pca[:, 1], c=preprocessed_df['kmeans_cluster'], cmap='viridis')

    for i, country in enumerate(countries):
        ax.text(numeric_data_pca[i, 0], numeric_data_pca[i, 1], str(country), fontsize=8, alpha=0.7)

    ax.set_title(f'K-means pentru {n_clusters} clustere')
    ax.set_xlabel("Componenta principală 1")
    ax.set_ylabel("Componenta principală 2")
    fig.colorbar(scatter, label='Cluster')
    st.pyplot(fig)

    return kmeans.labels_


# ==================== GEOPANDAS ====================

def geo_pandas_starting_section(europe, merged):
    st.write("Date geografice Europa, preluate prin GeoPandas de la Natural Earth:")
    st.dataframe(europe)

    st.write("Datele geopandas și indicatorii după join:")
    st.dataframe(merged)

    indicator = st.selectbox("Selectează indicator pentru hartă", options=[
        "life_expectancy_at_birth",
        "mortality_rate_infant",
        "current_health_expenditure",
        "total_air_emissions",
        "self_reported_unmet_needs_for_medical_examination",
        "total_unemployment_rate",
        "hipc",
        "self_perceived_health",
        "at_risk_of_poverty_rate"
    ])

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    merged.plot(column=indicator, cmap='viridis', legend=True, ax=ax,
                missing_kwds={"color": "lightgrey", "label": "No data"})

    ax.set_title(f"Harta {indicator.replace('_', ' ').capitalize()} ")
    ax.axis('off')
    ax.set_xlim(-20, 45)
    ax.set_ylim(30, 78)

    st.pyplot(fig)

def preprocess_data(df, cols):
    df_clean = df.copy()
    df_clean[cols] = df_clean[cols].fillna(df_clean[cols].mean())
    outliers_columns = [col for col in cols if df_clean[col].max() > 100]
    for col in outliers_columns:
        df_clean[col + "_log"] = np.log1p(df_clean[col])
    df_clean.drop(columns=outliers_columns, inplace=True)
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    df_std = pd.DataFrame(scaler.fit_transform(df_clean[numeric_cols]), columns=numeric_cols, index=df.index)
    minmax = MinMaxScaler()
    df_norm = pd.DataFrame(minmax.fit_transform(df_std), columns=numeric_cols, index=df.index)

    return df_norm


def afiseaza_harta_clusterelor(labels, data):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    europe = world[world['continent'] == 'Europe']
    df_clusters = data[['country']].copy()
    df_clusters['kmeans_cluster'] = labels
    merged = europe.set_index('name').join(df_clusters.set_index('country'))
    fig, ax = plt.subplots(figsize=(12, 8))
    merged.plot(column='kmeans_cluster', categorical=True, legend=True, cmap='tab10', ax=ax,
                missing_kwds={"color": "lightgrey", "label": "Fără date"})
    ax.set_title("Harta clusterelor KMeans")
    ax.axis('off')
    ax.set_xlim(-20, 45)
    ax.set_ylim(30, 78)
    st.pyplot(fig)