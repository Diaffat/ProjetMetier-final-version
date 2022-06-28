from streamlit_option_menu import option_menu
import streamlit as st,os
import dataexploration
import plots, pickle,pyodbc,time,datetime,base64,json,seaborn as sns,folium,requests,geopy,geocoder,numpy as np,pandas as pd,ipywidgets,streamlit as st ,matplotlib.pyplot as plt ,plotly.express as px,plotly.graph_objects as go
import runpredictions
from PIL import Image
import numpy as np,St_classification,dataframefunctions
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling
from streamlit_folium import folium_static
from streamlit_lottie import st_lottie,st_lottie_spinner
import sweetviz as sv

def predict_input(model):
    st.header(" Saisir les donner d'entrées")
    PH = st.number_input("PH")
    T = st.number_input("T")
    CE = st.number_input("CE")
    O2 = st.number_input("O2")
    NH = st.number_input("NH")
    NO = st.number_input("NO")
    SO = st.number_input("SO")
    PO = st.number_input("PO")
    DBO5 = st.number_input("DBO5")
    val = [PH, T, CE,	O2,	NH,	NO,	SO,	PO,	DBO5]
    ypred = model.predict([val])

    if ypred[0]==1:
        classe = "Excellente"
    if ypred[0]==2:
        classe = "Bonne"
    if ypred[0]==3:
        classe = "Peu Polluée"
    if ypred[0]==4:
        classe = "Mauvaise"
    if ypred[0]==5:
        classe = "Très mauvaise"
    v1,v2 = st.columns(2)
    #v1= st.write('Go')
    valider = v1.button("Valider")
    if valider:
        st.success(f"La qualité de ton eau est: {classe}")
     


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
c="",
def loti(c) :
    lottie_coding = load_lottiefile(c)  # replace link to local lottie file
          
    st_lottie(
              lottie_coding,
              speed=1,
              reverse=False,
              loop=True,
              quality="high", # medium ; high
         #renderer="svg", # canvas
              height=None,
              width=200,
              key=None,
               )  
def get_table_download_link(df,filename,text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download Report</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href
st.markdown("""
<style>
.first_titre {
    font-size:40px !important;
    font-weight: bold;
    box-sizing: border-box;
    text-align: center;
    width: 100%;
}
.intro{
    text-align: justify;
    font-size:20px !important;
}
.grand_titre {
    font-size:20px !important;
    font-weight: bold;
    text-align: center;
    # text-decoration: underline;
    text-decoration-color: #4976E4;
    text-decoration-thickness: 5px;
}
.section{
    font-size:20px !important;
    font-weight: bold;
    text-align: center;
    text-decoration: underline;
    text-decoration-color: #111111;
    text-decoration-thickness: 3px;
}
.petite_section{
    font-size:16px !important;
    font-weight: bold;
}
.nom_colonne_page3{
    font-size:17px !important;
    text-decoration: underline;
    text-decoration-color: #000;
    text-decoration-thickness: 1px;
}
</style>
""", unsafe_allow_html=True)

# def load_data():
#     if "dataframe" not in st.session_state:
#             try:
#                 if 'csv' in st.session_state.file_details['FileName']:
#                     if st.session_state.separateur != "":
#                         st.session_state.dataframe = pd.read_csv(uploaded_file, sep=st.session_state.separateur, engine='python')
#                     else:
#                         st.session_state.dataframe = pd.read_csv(uploaded_file)
#                 else:
#                     if st.session_state.separateur != "":
#                         st.session_state.dataframe = pd.read_excel(uploaded_file, sep=st.session_state.separateur, engine='python')
#                     else:
#                         st.session_state.dataframe = pd.read_excel(uploaded_file)
#             except:
#                 pass






choose = option_menu(None,["Home","Prepocesing","Vizualisation","M.Learning",'Prediction',"Rapport",'MAP Moulouya',],
    icons=['house',"bi bi-file-bar-graph",'bi bi-search','graph-up','',"bi bi-cloud-check","bi bi-geo-alt"],
    menu_icon = "None", default_index=0,
    styles={
        "container": {"padding": "5!important", "background-color": ""},
        "icon": {"color": "orange", "font-size": "18px"}, 
        "nav-link": {"font-size": "10px", "text-align": "left", "margin":"5px", "--hover-color": ""},
        "nav-link-selected": {"background-color": ""},
    },orientation = "horizontal"
    )
#*********************************************************************
if choose=="Home":
    st.markdown('<p class="first_titre">Machine Learning Platform</p>', unsafe_allow_html=True)
    st.write("---")
    c1, c2 = st.columns((3, 2))
    with c1:
        st.write("##")
        st.markdown(
            '<p class="intro"><b>Bienvenue sur notre plateforme de data science realisée par: BAGUIAN HAROUNA ET DIASSANA FATOUMATA!</b></p>',
            unsafe_allow_html=True)
    with c1:
        st.subheader("Suivez nous sur Github:")
        st.write(
            "• [DIASSANA Fatoumata/GitHub](https://github.com/Diaffat)")
    # if 'a' in st.session_state:  
    #         st.session_state['a'] = "Bonjour"
    # st.write(st.session_state.a)
    # df = pd.read_csv("df")
    # pr = df.profile_report()

    # st_profile_report(pr)
# if choose=="Prediction":
#     c=list(st.session_state.listeM)
#     modelpredict = st.selectbox("Model of prediction",reversed(c))
#     modelpredict=str(modelpredict)
#     m=pickle.load(open("ProjetMetierV2\modelesTrain"+modelpredict,"rb"))
#     predict_input(m)
if choose=="Prediction":
    c=list(st.session_state.listeM)
    listez = os.listdir('ProjetMetierV2\modelesTrain')

    kind = st.selectbox("Kind of predict",["one prediction","Multiprediction"])
    if kind=="one prediction":
        modelpredict = st.selectbox("Model of prediction",listez)
        m=pickle.load(open("ProjetMetierV2\modelesTrain/"+modelpredict,"rb"))
        v1,v2 = st.columns(2)
        predict_input(m)
        
    if kind=="Multiprediction":
        with st.sidebar:
            # st.markdown("## *1.First Step* ##")
            data = st.sidebar.file_uploader("Please upload your dataset (CSV format):", type=['csv','xlsx'])
            is_loaded_dataset = st.sidebar.warning("Dataset not uploaded")
            if data is not None:
                is_loaded_dataset.success("Dataset uploaded successfully!")
                data = pd.read_csv(data)
                #data.to_csv("dataprediction")
        modelpredict = st.selectbox("Model of prediction",listez)
        m=pickle.load(open("ProjetMetierV2\modelesTrain/"+modelpredict,"rb"))
        St_classification.predict_inputm(m)
      
if choose == "MAP Moulouya":
           st.title("Ci-dessous une vue des differentes stations du fleuve MOULOUYA")
           markers_dict = {"Ait boulmane": [ 31.0 , -7.1], 
                            "Source Arbalou": [31.2911, -9.2391], 
                            "Boumia": [32.7253, -5.1018], 
                            "Zaïda": [32.8026, -4.8709], 
                            "AnzarOufounas": [33.1633, -5.1638],
                            "Tamdafelt":[32.8722, -4.2564], 
                            "Missour":[33.0471, -3.9926],  
                            "Sebra":[34.826,-1.5303],
                            "Safsaf":[34.9139,-2.6225], 
                            "Pont Hassan II":[34.0254,-6.8222],
                            "Moulouya":[35.1092,-2.3578]}
           # create map
           map_cities = folium.Map(location=[31.7, -11.6],zoom_start=4)
           # plot locations
           for i in markers_dict.items():
               folium.Marker(location=i[1], popup=i[0]).add_to(map_cities)
               print(i)
            # display map
           folium_static(map_cities)           

if choose=="Prepocesing":
    # with st.sidebar:
    #     st.markdown("## **1.First Step** ##")
    #     data = st.sidebar.file_uploader("Please upload your dataset (CSV format):", type=['csv','xlsx'])
    #     is_loaded_dataset = st.sidebar.warning("Dataset not uploaded")
    #     if data is not None:
    #         is_loaded_dataset.success("Dataset uploaded successfully!")
    
    st.session_state.data = dataframefunctions.get_dataframe()
    if st.session_state.data is not None:
        dataexploration.load_page(st.session_state.data)

    # df = dataframe.copy()
    # df.to_csv("df")
        
    # if data is not None:

    #     dataexploration.load_page(df)
            
            
                # # is_loaded_dataset.success("Dataset uploaded successfully!")
                # dataframe = dataframefunctions.get_dataframe(st.session_state.dataframe)
                
    # if st.session_state.dataframe is not None:
    #     dataexploration.load_page(df)
    
#*********************************************************************
if choose=="Vizualisation":
    
    df = pd.read_csv("df")
    select = st.sidebar.selectbox("Select the Kind of plot",["Plot","Statistique Prediction"])
    if select== "Plot":
        plots.load_page(df)

if choose=="M.Learning":
    df = pd.read_csv("df")
    select1 = st.sidebar.selectbox("Choose",["Classification","Regression"])
    if select1=="Regression":
        runpredictions.load_page(df)
    if select1=="Classification":
        dataframe=pd.read_csv('data')
        St_classification.load(dataframe)
connection = pyodbc.connect(r'DRIVER={ODBC Driver 17 for SQL Server};'
    r'SERVER=DESKTOP-PTQ7CTJ\SQLEXPRESS;'
    r'DATABASE=DataProjetMetier;'
    r'Trusted_Connection=yes;') 
cursor = connection.cursor()
  

if choose=="Rapport":
    st.success("Welcome")
    # Display Data
    cursor.execute('''SELECT*FROM DUsers''')
    data = cursor.fetchall()
    st.header("**User's👨‍💻 Data**")
    df = pd.read_sql_query('''SELECT*FROM DUsers''',connection)
    st.dataframe(df)
    st.markdown(get_table_download_link(df,'MODELES.csv','Download Report'), unsafe_allow_html=True)
    cursor.execute('''SELECT*FROM DUsers''')
    data = cursor.fetchall()
    query = 'select * from DUsers;'
    plot_data = pd.read_sql(query, connection)
    ## Pie chart for predicted field recommendations
    labels = plot_data.ACCURACY.unique()
    print(labels)
    values = plot_data.ACCURACY.value_counts()
    print(values)
    st.subheader("📈 **ACCCURACY**")
    fig = px.pie(df, values=values, names=labels, title='Accuracy')
    st.plotly_chart(fig)
    labels = plot_data.PRECISION.unique()
    values = plot_data.PRECISION.value_counts()
    st.subheader("📈 ** Pie-Chart for modele's👨‍💻 precision**")
    fig = px.pie(df, values=values, names=labels, title="Pie-Chart📈 for modele's👨‍💻 precision")
    st.plotly_chart(fig)
    ### Pie chart for User's👨‍💻 Experienced Level
    labels = plot_data.hyperpara.unique()
    values = plot_data.hyperpara.value_counts()
    st.subheader("📈 ** Pie-Chart for modele's👨‍💻 hyperparameters**")
    fig = px.pie(df, values=values, names=labels, title="Pie-Chart📈 for modele's👨‍💻 hyperparametres")
    st.plotly_chart(fig)

