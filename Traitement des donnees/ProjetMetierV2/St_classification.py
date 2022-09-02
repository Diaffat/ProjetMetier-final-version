from unicodedata import numeric
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree
import numpy as np,pyodbc,time,datetime,base64
import streamlit as st
from sklearn.metrics import classification_report
import runpredictions
import pandas as pd
import plotly.graph_objects as go
import pickle
def download_model(model,nomModele):
  with open('ProjetMetierV2\modelesTrain'+nomModele, 'wb') as f:
    output_model = pickle.dump(model, f)
  b64 = base64.b64encode(output_model).decode()
  href = f'<a href="data:file/output_model;base64,{b64}">Download Trained Model .pkl File</a> (right-click and save as &lt;some_name&gt;.pkl)'
  st.markdown(href, unsafe_allow_html=True)

listeM=[]
if 'listeM' not in st.session_state:
  st.session_state.listeM =[]
def predict_inputm(model):
    st.header(" Saisir les donner d'entrées")
    dataprediction = pd.read_csv("data")
    columns = dataprediction.columns.to_list()
    PH = st.selectbox("PH",columns)
    TE = st.selectbox("T",columns)
    CE = st.selectbox("CE",columns)
    O2 = st.selectbox("O2",columns)
    NH = st.selectbox("NH",columns)
    NO = st.selectbox("NO",columns)
    SO = st.selectbox("SO",columns)
    PO = st.selectbox("PO",columns)
    DBO5 = st.selectbox("DBO5",columns)
    classes = []
    for i in range(dataprediction.shape[0]):
        val = [dataprediction[PH][i],dataprediction[TE][i], dataprediction[CE][i], dataprediction[O2][i],dataprediction[NH][i],dataprediction[NO][i],dataprediction[SO][i],dataprediction[PO][i],dataprediction[DBO5][i]]
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
        classes.append(classe)
    df = dataprediction.assign(classes = classes)
    st.write(df)

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
    st.success(f"La qualité de ton eau est: {classe}")
connection = pyodbc.connect(r'DRIVER={ODBC Driver 17 for SQL Server};'
    r'SERVER=DESKTOP-PTQ7CTJ\SQLEXPRESS;'
    r'DATABASE=DataProjetMetier;'
    r'Trusted_Connection=yes;') 
cursor = connection.cursor()

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

def insert_data(nomModele,hyperpara,PRECISION,RECALL_SCORE,F1_SCORE,ACCURACY,TIMESTAMPS):
    DB_table_name = 'DUsers'
    insert_sql = "insert into " + DB_table_name + """
    values (?,?,?,?,?,?,?)"""
    rec_values = (nomModele,hyperpara,PRECISION,RECALL_SCORE,F1_SCORE,ACCURACY,TIMESTAMPS)
    cursor.execute(insert_sql, rec_values)
    connection.commit()
def load(dataframe):
        numeric_column = dataframe.select_dtypes(exclude=("object")).columns
        st.title("Classification")
        col1,col2= st.columns(2)
        model = col1.selectbox("Model",["--Choose Model--","SVC","Tree","KNN","RandomFrorest","Voting","Bagging"])
        # col3,2col2= st.columns(2)
        # gridsearch = col3.checkbox("GridSearch")
        # randomsearch = col4.checkbox("RandomSearch")
        option = col2.radio("Navigation",["Auto","GridSearch"],horizontal=True)

        colf,colt= st.columns(2)
        st.sidebar.title("Select features")
        feature = st.sidebar.multiselect("Features",numeric_column)
        st.sidebar.title("Select target")
        target = st.sidebar.selectbox("target",numeric_column)
        x = dataframe[feature]
        y = dataframe[target]
        colx,coly= st.columns(2)
        st.sidebar.title("Choose Test size")
        test_size = st.sidebar.slider("Test size",min_value=1,max_value=5,help="Choose the test size")
        x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=test_size/10)


        ##############################metric###############################

        def metrics(data,model):
                model.fit(x_train, y_train)
                if len(y.unique()) >=2:
                    st.markdown('<p class="section">METRICS</p>',unsafe_allow_html=True)
                    col1,col2,col3,col4 = st.columns(4)
                    y_pred_test = model.predict(x_test.values)
                    y_pred_train = model.predict(x_train.values)
                    # test
                    precision = precision_score(y_test, y_pred_test, average='macro')
                    recall = recall_score(y_test, y_pred_test, average='weighted')
                    f1 = f1_score(y_test, y_pred_test, average='macro')
                    acc = accuracy_score(y_test, y_pred_test)
                    # train
                    precis_train = precision_score(y_train, y_pred_train, average='macro')
                    rappel_train = recall_score(y_train, y_pred_train, average='macro')
                    F1_train = f1_score(y_train, y_pred_train, average='macro')
                    accur_train = accuracy_score(y_train, y_pred_train)
                    # metrics
                    col1.metric(label="Accuracy", value=round(acc, 3),delta=round(acc - accur_train, 3))
                    col2.metric(label="F1 score", value=round(f1, 3),delta=round(f1 - F1_train, 3))
                    col3.metric(label="Recall", value=round(recall, 3),delta=round(recall - rappel_train, 3))
                    col4.metric(label="Precision", value=round(precision, 3),delta=round(precision - precis_train, 3))
                    with st.expander('Metrics report'):
                      st.text('Model Report:\n ' + classification_report(y_test, model.predict(x_test)))
                    st.markdown('<p class="section">INFO Metrics</p>',
                             unsafe_allow_html=True)
                    with st.expander('Info Metrics'):
                       runpredictions.report()

                    st_learning_curve(model,x_train,y_train)
        ##############################st_learning_curvec###############################
        def st_learning_curve(model,x_train,y_train):
            st.write("##")
            st.markdown(
                '<p class="section">Learning curves</p>',
                unsafe_allow_html=True)
            st.write("##")
            N, train_score, val_score = learning_curve(model, x_train, y_train,train_sizes=np.linspace(0.2,1.0,10),cv=3, random_state=4)
            fig = go.Figure()
            fig.add_scatter(x=N, y=train_score.mean(axis=1), name='train',
                            marker=dict(color='deepskyblue'))
            fig.add_scatter(x=N, y=val_score.mean(axis=1), name='validation',
                            marker=dict(color='red'))
            fig.update_layout(
                showlegend=True,
                template='simple_white',
                font=dict(size=10),
                autosize=False,
                width=500, height=300,
                margin=dict(l=40, r=50, b=40, t=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig)

        ##############################st_svc_hypperparams###############################
        def st_svc_hypperparams():
                svc = SVC(C= 0.1, gamma=1, kernel='linear')
                st.header("SVC Hyper_Params")
                namec,firstc,midlec=st.columns(3)
                Min_value_C = firstc.number_input("Min_value C",value=0.01)
                Max_value_C = midlec.number_input("Max_value C",value=0.1)
                namec.header("C:")
                namegamma,firstgamma,midlegamma=st.columns(3)
                Min_value_g = firstgamma.number_input("Min_value Gamma",value=0.01)
                Max_value_g = midlegamma.number_input("Max_value Gamma",value=0.1)
                namegamma.header("GAMMA:")
                    
                name,last=st.columns(2)
                name.header("Kernel :")
                kernel = last.multiselect("Choose Kernel(s)",["sigmoid","poly","rbf"])
                v1,v2,v3 = st.columns(3)

                valider = v1.checkbox("Valider")
                #down=v3.button("Download model")
                save = v2.button("Save model")

                if valider:
                    params = {"C":np.arange(Min_value_C,Max_value_C,0.1),"gamma":np.arange(Min_value_g,Max_value_g,0.1),"kernel":kernel}
                    grid = GridSearchCV(svc,params)
                    grid.fit(x_train,y_train)
                    hyperpara=grid.best_params_
                    svc = grid.best_estimator_
                    metrics(dataframe,svc)
                if save:
                  filename = "svc"
                  pickle.dump(svc,open(filename,"wb"))
                  st.write("model saved")
                  precision = precision_score(y_test, svc.predict(x_test), average='macro')
                  recall = recall_score(y_test, svc.predict(x_test), average='weighted')
                  f1 = f1_score(y_test, svc.predict(x_test), average='macro')
                  acc = accuracy_score(y_test, svc.predict(x_test))
                  ts = time.time()
                  cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                  cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                  timestamp = str(cur_date+'/'+cur_time)
                  nomModele=filename +'_'+ timestamp
                  nomModele=str(filename +'_'+ str(round(acc,2)))
                  with open('ProjetMetierV2\modelesTrain'+nomModele, 'wb') as f:
                      p=pickle.dump(svc, f)
                  hyperpara=str(hyperpara)
                  insert_data(nomModele,hyperpara,str(precision),str(recall),str(f1),str(acc),timestamp)
                  st.balloons()
                  st.session_state.listeM.append(nomModele)
                      
                #if down:
                  #download_model(svc,nomModele)
                  #st.balloons()
                  #st.markdown(get_table_download_link(df,'Modele'+nomModele,'Download modeles'), unsafe_allow_html=True)    


        def st_KNN_hypperparams():

            mknn =  KNeighborsClassifier(n_neighbors=1)
            mknn.fit(x_train, y_train)
            st.header("KNN Params")
            kcl1,kcl2,kcl3=st.columns(3)

            kcl1.header("n_neighbors")
            k=kcl2.number_input("choose min neighbors")
            kmax=kcl3.number_input("choose max neighbors")
            wcl1,wcl2=st.columns(2)
            wcl1.header("weight")
            l= wcl2.multiselect("Choose weight",["uniform","distance"])
            acl1,acl2=st.columns(2)

            acl1.header("    Algorithm")
            ac=acl2.multiselect("Choose algo",["ball_tree","auto","kd_tree"])
            v1,v2 = st.columns(2)

            valider = v1.checkbox("Valider")

            save = v2.button("Save model")

            if valider:
                params = {"n_neighbors":np.arange(int(k),int(kmax),1),"weights":l,"algorithm":ac}
                grid = GridSearchCV(mknn,params)
                grid.fit(x_train,y_train)
                hyperpara = grid.best_params_
                mknn = grid.best_estimator_
                metrics(dataframe,mknn)
            if save:
                filename = "knn"
                precision = precision_score(y_test,mknn.predict(x_test), average='macro')
                recall = recall_score(y_test, mknn.predict(x_test), average='weighted')
                f1 = f1_score(y_test, mknn.predict(x_test), average='macro')
                acc = accuracy_score(y_test, mknn.predict(x_test))
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'/'+cur_time)
                nomModele=filename +'_'+ timestamp
                acc=round(acc,2)
                nomModele=str(filename +'_'+ str(acc))
                with open('ProjetMetierV2\modelesTrain'+nomModele, 'wb') as f:
                    pickle.dump(mknn, f)
                st.write("model saved")
                hyperpara=str(hyperpara)
                insert_data(nomModele,hyperpara,str(precision),str(recall),str(f1),str(acc),timestamp)
                st.session_state.listeM.append(nomModele)
                st.balloons()        


        def st_Tree_hypperparams():

                mtree = tree.DecisionTreeClassifier(ccp_alpha=0.1, max_depth=3, min_samples_split=2, min_weight_fraction_leaf=0.1)
                mtree.fit(x_train, y_train)

                st.header("Tree Params")
                tcl1,tcl2,tcl3=st.columns(3)

                tcl1.header("min_impurity")
                t=tcl2.number_input("choose min value decrease")
                tmax=tcl3.number_input("choose max  value decrease")
                trl1,trcl2,trcl3=st.columns(3)
                trl1.header("min-leaf ")
                trs= trcl2.number_input("choose min value ")
                tres= trcl3.number_input("choose max value ")
                trls1,trcls2,trcls3=st.columns(3)
                trls1.header("min_split")
                tr= trcls2.number_input("choose min value split ")
                tre= trcls3.number_input("choose max value split ")
                tr1,tr2,tr3=st.columns(3)
                tr1.header("ccp alpha")
                trf= tr2.number_input("choose min value ccp ")
                trf2= tr3.number_input("choose max value ccp")
                tri1,tri2=st.columns(2)
                tri1.header("max features")
                trfi= tri2.multiselect("Choose one ",["auto", "sqrt", "log2","None"] )
                v1,v2 = st.columns(2)

                valider = v1.checkbox("Valider")

                save = v2.button("Save model")

                if valider:
                    params = {"min_impurity_decrease":np.arange(t,tmax,0.1),"min_samples_split":np.arange(tr,tre,0.1),"min_samples_leaf":np.arange(trs,tres,0.1),"ccp_alpha":np.arange(trf,trf2,0.1),"max_features":trfi}
                    grid = GridSearchCV(mtree,params)
                    grid.fit(x_train,y_train)
                    hyperpara =grid.best_params_
                    mtree = grid.best_estimator_
                    metrics(dataframe,mtree)
                if save:
                    filename = "tree"
                    precision = precision_score(y_test,mtree.predict(x_test), average='macro')
                    recall = recall_score(y_test, mtree.predict(x_test), average='weighted')
                    f1 = f1_score(y_test, mtree.predict(x_test), average='macro')
                    acc = accuracy_score(y_test, mtree.predict(x_test))
                    ts = time.time()
                    cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    timestamp = str(cur_date+'/'+cur_time)
                    nomModele=filename +'_'+ timestamp
                    acc=round(acc,2)
                    nomModele=str(filename +'_'+ str(acc))
                    with open('ProjetMetierV2\modelesTrain'+nomModele, 'wb') as f:
                       pickle.dump(mtree, f)
                    st.write("model saved")
                    hyperpara=str(hyperpara)
                    insert_data(nomModele,hyperpara,str(precision),str(recall),str(f1),str(acc),timestamp)
                    st.session_state.listeM.append(nomModele)
                    st.balloons()        

        def st_RandomForest_hypperparams():
                raf = RandomForestClassifier(n_estimators=100)
                raf.fit(x_train, y_train)

                st.header("RandomFrorest hyperParams")
                tcl1,tcl2,tcl3=st.columns(3)

                tcl1.header("n_estimators")
                ta=tcl2.number_input("choose min value ",min_value=10, max_value=100, value=10)
                tmaxa=tcl3.number_input("choose max  value ",min_value=10, max_value=100, value=20)
                trl1,trcl2,trcl3=st.columns(3)
                # trl1.header("max_depth")
                # trsa= trcl2.number_input("choose mini value ")
                # tresa= trcl3.number_input("choose maxi value ")
                trls1,trcls2,trcls3=st.columns(3)
                trls1.header("min_split")
                tra= trcls2.number_input("choose min value split ")
                trea= trcls3.number_input("choose max value split ")
                tr1,tr2,tr3=st.columns(3)
                tr1.header("ccp alpha")
                trfa= tr2.number_input("choose min value ccp ")
                trf2a= tr3.number_input("choose max value ccp")
                tri1,tri2=st.columns(2)
                tri1.header("max features")
                trfia= tri2.multiselect("Choose one ",["auto", "sqrt", "log2","None"] )
                v1,v2 = st.columns(2)

                valider = v1.checkbox("Valider")

                save = v2.button("Save model")

                if valider:
                    params = {"n_estimators":np.arange(int(ta),int(tmaxa)),"min_samples_leaf":np.arange(tra,trea,0.1),"ccp_alpha":np.arange(trfa,trf2a,0.1),"max_features":trfia}
                    grid = GridSearchCV(raf,params)
                    grid.fit(x_train,y_train)
                    hyperpara=grid.best_params_
                    raf = grid.best_estimator_
                    metrics(dataframe,raf)
                if save:
                    filename = "RAF"
                    precision = precision_score(y_test,raf.predict(x_test), average='macro')
                    recall = recall_score(y_test, raf.predict(x_test), average='weighted')
                    f1 = f1_score(y_test, raf.predict(x_test), average='macro')
                    acc = accuracy_score(y_test, raf.predict(x_test))
                    ts = time.time()
                    cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    timestamp = str(cur_date+'/'+cur_time)
                    acc=round(acc,2)
                    nomModele=str(filename +'_'+ str(acc))
                    with open('ProjetMetierV2\modelesTrain'+nomModele, 'wb') as f:
                       pickle.dump(raf, f)
                    st.write("model saved")
                    hyperpara=str(hyperpara)
                    insert_data(nomModele,hyperpara,str(precision),str(recall),str(f1),str(acc),timestamp)
                    st.session_state.listeM.append(nomModele)
                    st.balloons()        
                
        svc = SVC()
        raf = RandomForestClassifier(n_estimators=100)
        mknn = KNeighborsClassifier(n_neighbors=1)
        mtree = tree.DecisionTreeClassifier(ccp_alpha=0.1, max_depth=3, min_samples_split=2, min_weight_fraction_leaf=0.1)
        if option=="Auto" and model=="Tree":
          metrics(dataframe,svc)

        elif option=="Auto" and model=="KNN":
          metrics(dataframe,mknn)

        elif option=="Auto" and model=="SVC":
          metrics(dataframe,mknn)

        elif option=="Auto" and model=="RandomFrorest":
          metrics(dataframe,raf)
        if option=="GridSearch" and model=="Tree":
          st_Tree_hypperparams()

        elif option=="GridSearch" and model=="SVC":
          st_svc_hypperparams()

        elif option=="GridSearch" and model=="KNN":
          st_KNN_hypperparams()

        elif option=="GridSearch" and model=="RandomFrorest":
            st_RandomForest_hypperparams()

      # return {'listeM':listeM }