import streamlit as st
import dataframefunctions
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from numpy import percentile




POSSIBLE_ATTRIBUTES_EXPLORATIONS = ["Scatter plot",
                                    "Box plot",
                                    "Correlation matrix",
                                    "Count plot",
                                    "Distribution plot"]


def load_page(dataframe):
    attrexp_action = st.selectbox("Select the method", POSSIBLE_ATTRIBUTES_EXPLORATIONS)

    if attrexp_action == "Scatter plot":
        render_scatterplot(dataframe)

    elif attrexp_action == "Box plot":
        render_boxplot(dataframe)

    elif attrexp_action == "Correlation matrix":
        render_corr_matrix(dataframe)

    elif attrexp_action == "Count plot":
        render_count_plot(dataframe)

    elif attrexp_action == "Distribution plot":
        render_distplot(dataframe)


def render_scatterplot(dataframe):
    """Renders a scatterplot based on the user's input."""

    df_columns = list(dataframe.columns.values)
    label_name = df_columns[len(df_columns) - 1]

    first_attribute = st.selectbox('Which feature on x?', df_columns)
    second_attribute = st.selectbox('Which feature on y?', df_columns, index=2)
    alpha_value = st.sidebar.slider('Alpha', 0.0, 1.0, 1.0)
    colored = st.sidebar.checkbox("Color based on Label", value=True)
    sized = st.checkbox("Size based on other attribute", value=False)
    if sized:
        size_attribute = st.selectbox('Which attribute?', dataframefunctions.get_numeric_columns(dataframe))

    with st.spinner("Plotting data.."):
        fig = px.scatter(dataframe,
                         x=first_attribute,
                         y=second_attribute,
                         color=label_name if colored else None,
                         opacity=alpha_value,
                         size=None if not sized else size_attribute)

        st.plotly_chart(fig)


def render_boxplot(dataframe):
    """Renders a boxplot based on the user's input."""
    
    def outliers(dataframe,ft):
        Q1, Q3 = percentile(dataframe[ft], 25), percentile(dataframe[ft], 75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        ls = dataframe.index[(dataframe[ft]<lower) | (dataframe[ft]>upper)]
        return ls
    def remove(dataframe,ls):
        ls = sorted(set(ls))
        dataframe = dataframe.drop(ls)
        return dataframe

    
    categorical_columns = dataframefunctions.get_categorical_columns(dataframe)
    numeric_columns = dataframefunctions.get_numeric_columns(dataframe)
    Numeric_columns = dataframe.select_dtypes(exclude=['object']).columns
    # st.write(numeric_columns)

    # boxplot_att1 = st.selectbox('Which feature on x? (only categorical features)', categorical_columns)
    boxplot_att2 = st.multiselect('Which feature on y? (only numeric features)',Numeric_columns)
                                # index=len(numeric_columns) - 1)
    show_points = st.sidebar.checkbox("Show points", value=False)
    


    with st.spinner("Plotting data.."):
        df = dataframe[boxplot_att2]
        
        fig = px.box(dataframe,
                    # x=boxplot_att1,
                    y=boxplot_att2,
                    points='all' if show_points else 'outliers')


        st.plotly_chart(fig)
        
        if len(boxplot_att2)!=0:
            st.header("Outliners")
            Features_list = dataframe.columns.to_list()
            Features_list
            index_list = []
            for feature in Numeric_columns:
                index_list.extend(outliers(dataframe,feature))

            st.multiselect("Outliers Indexs",sorted(set(index_list)),default=sorted(set(index_list)))

            
            if st.button("Remove"):
            
                dataframe = remove(dataframe,index_list)
                st.success('Suppresion successfully')
                st.write(dataframe.shape)
                st.session_state.data = dataframe


    # if st.checkbox("Remove"):
    #     Q1 = dataframe[boxplot_att2].quantile(0.25)
 
    #     Q3 = dataframe[boxplot_att2].quantile(0.75)
    #     IQR = Q3 - Q1
    #     inf = Q1 - 1,5*IQR
    #     sup = Q3 + 1,5*IQR
    #     dataframe[boxplot_att2] = dataframe[dataframe[boxplot_att2]<inf]
    #     dataframe[boxplot_att2] = dataframe[dataframe[boxplot_att2]>sup]
    #     # # Upper bound
    #     # upper = np.where(dataframe[boxplot_att2] >= sup)
    #     # # Lower bound
    #     # lower = np.where(dataframe[boxplot_att2] <= inf)
    #     # ''' Removing the Outliers '''
    #     # dataframe.drop(upper[0], inplace = True)
    #     # dataframe.drop(lower[0], inplace = True)
        


def render_corr_matrix(dataframe):
    """Renders a correlation matrix based on the user's input."""

    if len(dataframefunctions.get_numeric_columns(dataframe)) > 30:
        st.warning("Warning: since the dataset has more than 30 features, the figure might be inaccurate.")

    corr = dataframe.corr()

    # Masking the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    f, ax = plt.subplots(figsize=(15, 12))
    cmap = sns.diverging_palette(10, 140, n=9, s=90, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

    with st.spinner("Plotting data.."):
        st.pyplot(f)


def render_count_plot(dataframe):
    """Renders a count plot based on the user's input."""
    f, ax = plt.subplots(figsize=(15, 12))

    feature = st.selectbox('Which feature?', list(dataframe.columns))
    with st.spinner("Plotting data.."):
        sns.countplot(x=feature, data=dataframe)
        st.pyplot(f)


def render_distplot(dataframe):
    """Renders a distribution plot based on the user's input."""
    f, ax = plt.subplots(figsize=(15, 12))

    feature = st.selectbox('Which feature?', dataframefunctions.get_numeric_columns(dataframe))
    with st.spinner("Plotting distribution.."):
        sns.distplot(dataframe[feature], color='g')
        st.pyplot(f)
    sapyro_test(dataframe[feature],0.05)


def sapyro_test(data,c):
    c=0.05
    pvalue = stats.shapiro(data)[1]
    st.write("le pvalue est de **%f**" %(pvalue))
    if pvalue<c:
        st.success("Les donnees ne suivent pas une loi normale")
    else:
        st.success("Les donnees suivent une loi normale")
