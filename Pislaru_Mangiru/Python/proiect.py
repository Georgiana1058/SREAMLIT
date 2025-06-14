from functii import *

data = pd.read_csv("Date/merged_data.csv")
variables = data.columns[1:]

section = st.sidebar.radio("Navigare", ["Vizualizarea Datelor", "Explorarea datelor", "Analiza EDA", "GeoPandas"])

if section == "Vizualizarea Datelor":
    vizualizare_date(data, variables)
elif section == "Explorarea datelor":
    explorare_date(data, variables)
elif section=="Analiza EDA":
    st.title("Analiza EDA")
    afiseaza_tipuri_si_nan(data)
    analiza_valori_lipsa(data)
    data = tratare_valori_lipsa(data)
    genereaza_histograme(data)
    afiseaza_distributie_variabila(data)
    analiza_corelatii(data)
    plot_pairplot_numeric(data)
    plot_boxplot_cat_numeric(data)
    outliers_columns = find_outliers_and_plot(data)
    data_log = apply_log_transform(data, outliers_columns)
    data_log.to_csv("data_log.csv")
    show_correlation_matrix(data_log)
    encoding_demo(data_log)
    data_std = standardizare_date(data_log)
    data_norm = normalizare_date(data_std)
    target = regresie_liniara_multipla(data_norm)
    corectie_model_regresie(data_norm, target)
    numeric_cols = data_norm.select_dtypes(include=['number']).columns.tolist()
    clusterizare_kmeans(data_norm, numeric_cols, data_log)
elif section=="GeoPandas":
    st.title("GeoPandas")
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    europe = world[world['continent'] == 'Europe']
    merged = europe.set_index('name').join(data.set_index('country'))

    geo_pandas_starting_section(europe, merged)
    data_processed = preprocess_data(data, variables)
    numeric_cols = data_processed.select_dtypes(include=['number']).columns.tolist()
    labels = clusterizare_kmeans(data_processed, numeric_cols, data)
    afiseaza_harta_clusterelor(labels, data)