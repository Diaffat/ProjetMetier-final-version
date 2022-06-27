from xml.etree.ElementInclude import include
import streamlit as st
import pandas as pd
import dataframefunctions
# import plots
import featuresanalysis
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plots
from collections import Counter
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling
import numpy as np
# import pycaret
import sweetviz as sv
import streamlit.components.v1 as components

POSSIBLE_DATAEXP_ACTIONS = ["Dataset Description","Preprocessing","Features Selection","Compare"]


def load_page(dataframe):
    st.sidebar.markdown("## **2. Second Step** ##")
    if dataframe is None:
        st.error("Please upload your dataset!")
    else:
        dataexp_action = st.sidebar.selectbox("What do you want to explore?", POSSIBLE_DATAEXP_ACTIONS)


    if dataexp_action=="Dataset Description":
        rdd = st.sidebar.radio("Choose",["First Look","Resume"])

        if rdd=="First Look":
            st.markdown("## **Exploring the dataset :mag:** ##")
            st.dataframe(dataframe.head(5))
            render_firstlook_comments(dataframe)
            st.markdown("## **Description :computer:** ##")
            st.dataframe(dataframe.describe())

        if rdd=="Resume":
            pr = dataframe.profile_report()
            st_profile_report(pr)

        #     render_first_look(dataframe)
        #     st.markdown("## **Description :computer:** ##")
        #     st.dataframe(dataframe.describe())
        #     # st.header("Infos")
        #     # st.dataframe(dataframe.info())

        # if rdd=="Analyse des colonnes":
        #     st.markdown('<p class="grand_titre">Analyse des colonnes</p>', unsafe_allow_html=True)
        #     st.write('##')
            
        #     options = dataframe.columns.to_list()
        #     slider_col = st.multiselect(
        #         'Selectionner une ou plusieurs colonnes',
        #         options, help="Choisissez les colonnes à analyser")
                
        #     col1, b, col2, c = st.columns((1.1, 0.1, 1.1, 0.3))
        #     with col1:
        #         st.write('##')
        #         st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
        #     with col2:
        #         st.write('##')
        #         st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
        #     for col in slider_col:
        #         ### Données ###
        #         data_col = dataframe[col].copy()
        #         n_data = dataframe[col].to_numpy()

        #         st.write('##')
        #         col1, b, col2, c = st.columns((1, 1, 2, 0.5))
        #         with col1:
        #             st.markdown('<p class="nom_colonne_page3">' + col + '</p>', unsafe_allow_html=True)
        #             st.write(data_col.head(20))
        #         with col2:
        #             st.write('##')
        #             st.write(' ● type de la colonne :', type(data_col))
        #             st.write(' ● type des valeurs :', type(data_col.iloc[1]))
        #             if n_data.dtype == float:
        #                 moyenne = data_col.mean()
        #                 variance = data_col.std()
        #                 max = data_col.max()
        #                 min = data_col.min()
        #                 st.write(' ● Moyenne :', round(moyenne, 3))

        #                 st.write(' ● Variance :', round(variance, 3))

        #                 st.write(' ● Maximum :', max)

        #                 st.write(' ● Minimum :', min)

        #             st.write(' ● Valeurs les plus présentes:', (Counter(n_data).most_common()[0])[0], 'apparait',
        #                     (Counter(n_data).most_common()[0])[1], 'fois', ', ', (Counter(n_data).most_common()[1])[0],
        #                     'apparait',
        #                     (Counter(n_data).most_common()[1])[1], 'fois')

        #             st.write(' ● Nombre de valeurs manquantes:',
        #                     sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist()))

        #             st.write(' ● Longueur:', n_data.shape[0])

        #             st.write(' ● Nombre de valeurs différentes non NaN:',
        #                     abs(len(Counter(n_data)) - sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist())))
        #             ### Fin section données ###
        #         st.write('##')

        

    if dataexp_action=="Compare": 

        st.markdown("## **Dataset before:computer:** ##")
        original = pd.read_csv("df") 
        st.write(original)
        comparison = pd.read_csv("data") 
        st.markdown("## **Dataset After:computer:** ##")
        st.write(comparison)
        st.download_button(data=comparison.to_csv(), label="Télécharger le dataset modifié",
                                   file_name='dataset.csv')
        sw = sv.compare([original, "Original"], [comparison, "Preprocessed"])
        sw.show_html(open_browser=False, layout='widescreen', scale=0.7) 
        display = open("SWEETVIZ_REPORT.html", 'r', encoding='utf-8')

        source_code = display.read()

        components.html(source_code, height=1200, scrolling=True)
       
        
                                                                                                                


    if dataexp_action=="Preprocessing": 
        
        st.sidebar.markdown("## **Null Values :computer:** ##")
        # rd = st.sidebar.radio("Choose",["Compute missing values","Compute linear correlation","Imputation Statistique"])
       
        # if rd=="Compute missing values":
        #     render_missing_data(dataframe)
        # if rd=="Compute linear correlation":
        #     render_linear_correlation(dataframe)


        # if rd=="Imputation Statistique":
        st.markdown("## **Missing values :mag:** ##")
        col1,col2 = st.columns(2)
        perc = col1.selectbox("Choose seuil NaN percentage",["100","90","80","70","60","50"])
        inputa1 = col2.selectbox("Numeric features Imputation",["--inputation--","Mean","Mediane","Zero"])
        # inputa2 = col3.selectbox("Categorical features Imputation",["--inputation--","Constant","Mode"])

        column_with_nan = dataframe.columns[dataframe.isnull().any()]
        dataframe.shape
        # for column in column_with_nan:
        #     print(column, dataframe[column].isnull().sum())
        
        c = []


        for column in column_with_nan:
                d = dataframe[column].isnull().sum()*100/dataframe.shape[0]
                st.write(d)
                if d > int(perc) :
                    dataframe = dataframe.drop(column,axis=1, inplace=True)

                    c.append(column)
                    # dataframe.to_csv("data")
        if inputa1=="Mean":
            numeric_columns = dataframe.select_dtypes(exclude=['object']).columns
            numeric_columns
            for col in numeric_columns:
                # print(np.mean(dataframe[col]))
                dataframe[col].fillna(round(np.mean(dataframe[col]),1),inplace=True)

            st.session_state.data = dataframe
            col2.markdown("## **Imputation :computer:** ##")
            if (dataframe[numeric_columns[0]].isnull().sum())==0:
                col2.success("Imputation successfully!")
                dataframe.to_csv("data")    
    
        col1.header("Droped Columns")
        col1.multiselect("Droped Columns", c, default=c)


        st.markdown("## **Outliners :computer:** ##")
        plots.render_boxplot(st.session_state.data)
        dataframe.to_csv("data")
        st.session_state.data = dataframe


        # col1.header("Dataset size")
        # col1.write(dataframe.shape)
        
        # if inputa=="Mean":
                



        # if rd=="Imputation Statistique":
        #     st.title("Replace NaN by :")
        #     col1,col2,col3,col4 = st.columns(4)
        #     col1.header("Mean ")
        #     col2.header("Mediane")
        #     col3.header("Mode")
        #     col4.header("Remove")
            # if col1.button("Remove NaN"):
            #     dataframe.dropna()
            #     st.dataframe(dataframe.describe( ))
            
            # col2.button("NaN by Mean")
            # col3.button("Result")
            # col4.button("Save")
            # col5.button("Dowload")
            


        # elif dataexp_action == "Plots":
        #     plots.load_page(dataframe)

        # elif dataexp_action == "Features":
        #     featuresanalysis.load_page(dataframe)

    # if dataexp_action=="Outliners":
    #     st.sidebar.markdown("## **Outliners :computer:** ##")
     
    #     plots.render_boxplot(dataframe)
    #     dataframe.to_csv("df")

    # if dataexp_action=="Normalization":
    #     st.sidebar.markdown("## **Normalization :computer:** ##")
    #     rd2 = st.sidebar.radio("Choose",["MinMax","Standard"])
    
    if dataexp_action=="Features Selection":
        st.markdown("## **Features Selection:computer:** ##")
        # st,st = st.columns(2)
        options = dataframe.columns.to_list()
        
        st.header("Drop columns")
        slider_col = st.multiselect('Selectionner une ou plusieurs colonnes',options, help="Choisissez les colonnes à supprimer")
        for col in slider_col:
            try:
                dataframe = dataframe.drop(columns=col, axis=1)
                
                st.success("Colonnes " + col + " supprimée !")
            except:
                st.error("Transformation impossible ou déjà effectuée") 
        dataframe.to_csv("data") 
        st.session_state.data = dataframe
        st.header("Encodage") 
        options2 = dataframe.select_dtypes(include=['object']).columns
        col_to_encodage = st.multiselect("Selectionner les colonnes à encoder",options2)
        for col in col_to_encodage:
            st.write("Colonne " + col + "  :  " + str(dataframe[col].unique().tolist()) + " -> " + str(np.arange(len(dataframe[col].unique()))))
            dataframe[col].replace(dataframe[col].unique(), np.arange(len(dataframe[col].unique())),inplace=True)
        dataframe.to_csv("data")
        st.session_state.data = dataframe
        render_linear_correlation(st.session_state.data)

        # st.sidebar.number_input("How many features",min_value=2,max_value=dataframe.shape[1])


def render_missing_data(dataframe):
    """Renders the missing values and the missing percentages for each column."""

    missing_values, missing_percentage = dataframefunctions.get_missing_values(dataframe)
    st.markdown("## **Missing values :mag:** ##")
    st.dataframe(pd.concat([missing_values, missing_percentage], axis=1, keys=["Total", "percent"]))


def render_first_look(dataframe):
    """Renders the head of the dataset (with nan values colored in red),
     and comments regarding instances, columns, and missing values."""

    number_of_rows = st.sidebar.slider('Number of rows', 1, dataframe.shape[0], 10)
    st.markdown("## **Exploring the dataset :mag:** ##")
    if st.sidebar.checkbox("Color NaN values in red", value=True):
        st.dataframe(dataframe.head(number_of_rows).style.applymap(dataframefunctions.color_null_red))
    else:
        st.dataframe(dataframe.head(number_of_rows))
    render_firstlook_comments(dataframe)
    # fig,ax =plt.subplots(figsize=(10,10))
    # st.dataframe(dataframe.dtypes.value_counts())
    # sns.heatmap(dataframe.isna())
    # st.pyplot(fig)


# TODO improve such that all the type of columns are considered
def render_firstlook_comments(dataframe):
    """Makes a first analysis of the dataset and shows comments based on that."""

    num_instances, num_features = dataframe.shape
    categorical_columns = dataframefunctions.get_categorical_columns(dataframe)
    numerical_columns = dataframefunctions.get_numeric_columns(dataframe)
    cat_column = categorical_columns[0] if len(categorical_columns) > 0 else ""
    num_column = numerical_columns[0] if len(numerical_columns) > 0 else ""
    total_missing_values = dataframe.isnull().sum().sum()

    st.write("* The dataset has **%d** observations and **%d** variables. \
             Hence, the _instances-features ratio_ is ~**%d**."
             % (num_instances, num_features, int(num_instances/num_features)))

    st.write("* The dataset has **%d** categorical columns (e.g. %s) and **%d** numerical columns (e.g. %s)."
             % (len(categorical_columns), cat_column, len(numerical_columns), num_column))

    st.write("* Total number of missing values: **%d** (~**%.2f**%%)."
             % (total_missing_values, 100*total_missing_values/(num_instances*num_features)))


def render_linear_correlation(dataframe):
    """If the label is not categorical, renders the linear correlation between the features and the label."""

    st.markdown("## **Linear correlation ** ##")
    df_columns = list(dataframe.columns.values)
    label_name = df_columns[len(df_columns) - 1]

    # If the label is not categorical, show an error
    if dataframefunctions.is_categorical(dataframe[label_name]):
        display_correlation_error()
        return

    positive_corr = dataframefunctions.get_linear_correlation(dataframe, label_name, positive=True)
    negative_corr = dataframefunctions.get_linear_correlation(dataframe, label_name, positive=False)
    st.write('Positively correlated features :chart_with_upwards_trend:', positive_corr)
    st.write('Negatively correlated features :chart_with_downwards_trend:', negative_corr)


def display_correlation_error():
    st.write(":no_entry::no_entry::no_entry:")
    st.write("It's **not** possible to determine a linear correlation with a categorical label.")
    st.write("For more info, please check [this link.]\
             (https://stackoverflow.com/questions/47894387/how-to-correlate-an-ordinal-categorical-column-in-pandas)")


# def methods_pyc(columns, model):
#     """
#     Define whwich imputation method to run on missing values
#     Define which features to ignore
#     Define miscellaneous methods
#     """

#     st.subheader("Missing values")
#     # sub_text("Select imputation methods for both numerical and categorical columns.")
#     imputation_num = st.selectbox("Select missing values imputation method for numerical features:",
#     ("mean", "median", "zero"))

#     imputation_cat = st.selectbox("Select missing values imputation method for categorical features:",
#     ("constant", "mode"))

#     sub_text("Select which columns to skip preprocessing for.")
#     ignore = st.multiselect("Select which columns to ignore:",
#     (columns), default = None)


def sweetviz_comparison(original, comparison, indicator, text, upload = True):

    """
    Function to compare test and train data with sweetviz
    """
    
        # call high level function and get files
    original =pd.read_csv("df.csv") 
    comparison = st.session_state.data

# use indicator to stop the app from running


    sw = sv.compare([original, "Original"], [comparison, "Comparison"])

    sw.show_html(open_browser=False, layout='vertical', scale=1.0)

    display = open("SWEETVIZ_REPORT.html", 'r', encoding='utf-8')

    source_code = display.read()

    components.html(source_code, height=1200, scrolling=True)

    create_pdf_html("SWEETVIZ_REPORT.html",
                    text,
                    "sweetviz_dqw.pdf")

    return(sw)
