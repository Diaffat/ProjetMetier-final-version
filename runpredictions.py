import sklearn
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from streamlit_lottie import st_lottie
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import plotly.graph_objects as go
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,confusion_matrix,roc_curve,roc_auc_score,auc
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt


POSSIBLE_MODEL = ["XGBoost (classifier)",
                  "XGBoost (regressor)",
                  "Random Forest",
                  "Support Vector Machine",
                  "K-nearest neighbors"]
KERNEL_OPTIONS = ['Rbf', 'Linear', 'Poly', 'Sigmoid']
EVALUATION_METRICS = ["Accuracy", "RMSE", "F1", "Precision", "Recall", "MSE"]
WEIGHT_FUNCTION_OPTION = ["Uniform", "Distance"]
ALGORITHM = ["Auto", "Ball tree", "Kd tree", "Brute"]
metrics2string = {"MSE": "neg_mean_squared_error",
                  "RMSE": "neg_mean_squared_error"}


def load_page(dataframe):
    """Loading the initial page, displaying the experiment parameters and the model's hyperparameters."""

    st.sidebar.subheader("Experiment parameters:")
    test_size = st.sidebar.slider('Test set size', 0.01, 0.99, 0.2)
    evaluation_metrics = st.sidebar.multiselect("Select the evaluation metrics", EVALUATION_METRICS)
    selected_model = st.sidebar.selectbox("Select the model", POSSIBLE_MODEL)
    cross_val = st.sidebar.checkbox("Cross validation")
    if cross_val:
        cv_k = st.sidebar.number_input("Please select the cross-validation K:",
                                       min_value=1,
                                       value=10,
                                       max_value=dataframe.shape[0])

    model_parameters = display_hyperparameters(selected_model)
    display_experiment_stats(dataframe, test_size, selected_model)
    if st.button("Run predictions"):
        if len(evaluation_metrics) == 0:
            st.error("Please select at least one evaluation metric!")
        else:
            run_predictions(dataframe,
                            test_size,
                            selected_model,
                            model_parameters,
                            evaluation_metrics,
                            cross_val,
                            cv_k if cross_val else None)


def run_predictions(dataframe, test_size, selected_model, parameters, metrics, cross_val, cv_k):
    """Puts together preprocessing, training and testing."""

    st.markdown(":chart_with_upwards_trend: Hyperparameters used: ")
    st.write(parameters)

    if cross_val:
        st.warning("Warning, only the first metric is selected when using Cross Validation.")

    # Preprocessing data
    x, y = preprocessing.preprocess(dataframe)
    st.success("Preprocessing completed!")

    model = get_model(selected_model, parameters)

    if cross_val:
        # model = get_model(selected_model, parameters)
        cross_validation(model, x, y, cv_k, metrics[0])

    else:
        # Training the model
        train_status = st.warning("Training model..")
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        # model = get_model(selected_model, parameters)
        model.fit(X_train, y_train)
        train_status.success("Training completed!")

        # Testing the model
        test_status = st.warning("Testing model..")
        test_model(model, X_train, y_train, X_test, y_test, metrics)
        test_status.success("Testing completed!")


def cross_validation(model, x, y, cv_k, metric):
    """Training and testing using cross validation."""

    current_status = st.warning("Training and testing model..")
    right_metric = metrics2string.get(metric, metric.lower())
    results = cross_validate(model, x, y, cv=cv_k, scoring=right_metric)
    scores = results.get("test_score")

    # When MSE or RMSE are used as metrics, the results are negative
    if metric == "MSE":
        scores = scores * -1

    if metric == "RMSE":
        scores = scores * -1
        scores = np.sqrt(scores)

    current_status.success("Training and testing completed!")
    evaluation = pd.DataFrame([[scores.mean(), scores.std()]],
                              columns=[[metric, metric], ["Mean", "Standard deviation"]],
                              index=["Dataset"])
    st.dataframe(evaluation)


def display_experiment_stats(dataframe, test_size, selected_model):
    """Displays the experiment input, e.g. test set size"""

    num_instances = dataframe.shape[0]
    training_instances = round(num_instances * (1 - test_size), 0)
    test_instances = round(num_instances * test_size, 0)
    st.write("Running **%s** with a test set size of **%d%%**." % (selected_model, round(test_size * 100, 0)))
    st.write("There are **%d** instances in the training set and **%d** instances in the test set." %
             (training_instances, test_instances))


def display_hyperparameters(selected_model):
    """Display the possible hyperparameters of the model chosen by the user."""

    hyperparameters = {}
    st.sidebar.subheader("Model parameters:")

    if selected_model == "XGBoost (classifier)":
        hyperparameters['learning_rate'] = st.sidebar.number_input('Learning rate',
                                                                   min_value=0.0001,
                                                                   max_value=10.0,
                                                                   value=0.1)
        hyperparameters['n_estimators'] = st.sidebar.slider("Num. estimators", 1, 500, 100)
        hyperparameters['max_depth'] = st.sidebar.slider("Maximum depth", 1, 20, 3)

    if selected_model == "XGBoost (regressor)":
        hyperparameters['learning_rate'] = st.sidebar.number_input('Learning rate',
                                                                   min_value=0.0001,
                                                                   max_value=10.0,
                                                                   value=0.1)
        hyperparameters['n_estimators'] = st.sidebar.slider("Num. estimators", 1, 500, 100)
        hyperparameters['max_depth'] = st.sidebar.slider("Maximum depth", 1, 100, 6)

    elif selected_model == "Random Forest":
        hyperparameters['n_estimators'] = st.sidebar.slider("Num. estimators", 1, 200, 100)
        hyperparameters['min_samples_split'] = st.sidebar.slider("Min. samples  split", 2, 20, 2)
        hyperparameters['criterion'] = st.sidebar.selectbox("Select the criteria", ['Gini', 'Entropy']).lower()
        hyperparameters['min_samples_leaf'] = st.sidebar.slider("Min. samples  leaf", 1, 50, 1)

    elif selected_model == "Support Vector Machine":
        hyperparameters['C'] = st.sidebar.number_input('Regularization', min_value=1.0, max_value=50.0, value=1.0)
        hyperparameters['kernel'] = st.sidebar.selectbox("Select the kernel", KERNEL_OPTIONS).lower()
        hyperparameters['gamma'] = st.sidebar.radio("Select the kernel coefficient", ['Scale', 'Auto']).lower()

    elif selected_model == "K-nearest neighbors":
        hyperparameters['n_neighbors'] = st.sidebar.slider("Num. neighbors", 1, 50, 5)
        hyperparameters['weights'] = st.sidebar.selectbox("Select the weight function", WEIGHT_FUNCTION_OPTION).lower()
        hyperparameters['algorithm'] = st.sidebar.selectbox("Select the algorithm", ALGORITHM).lower().replace(" ", "_")

    return hyperparameters


def get_model(model, parameters):
    """Creates and trains a new model based on the user's input."""

    if model == "XGBoost (classifier)":
        model = XGBClassifier(**parameters)

    elif model == "Random Forest":
        model = RandomForestClassifier(**parameters)

    elif model == "Support Vector Machine":
        model = SVC(**parameters)

    elif model == "K-nearest neighbors":
        model = KNeighborsClassifier(**parameters)

    elif model == "XGBoost (regressor)":
        model = XGBRegressor(**parameters)

    return model


def test_model(model, X_train, y_train, X_test, y_test, metrics):
    """Tests the model predictions based on the chosen metrics."""

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    metrics_data = {}

    if "RMSE" in metrics:
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        metrics_data["RMSE"] = [np.sqrt(train_mse), np.sqrt(test_mse)]

    if "Accuracy" in metrics:
        train_accuracy = accuracy_score(y_train, y_train_pred) * 100.0
        test_accuracy = accuracy_score(y_test, y_test_pred) * 100.0
        metrics_data["Accuracy"] = [train_accuracy, test_accuracy]

    if "F1" in metrics:
        f1_train = f1_score(y_train, y_train_pred, average='micro') * 100.0
        f1_test = f1_score(y_test, y_test_pred, average='micro') * 100.0
        metrics_data["F1-Score"] = [f1_train, f1_test]

    if "Precision" in metrics:
        precision_train = precision_score(y_train, y_train_pred, average='micro') * 100.0
        precision_test = precision_score(y_test, y_test_pred, average='micro') * 100.0
        metrics_data["Precision"] = [precision_train, precision_test]

    if "Recall" in metrics:
        recall_train = recall_score(y_train, y_train_pred, average='micro') * 100.0
        recall_test = recall_score(y_test, y_test_pred, average='micro') * 100.0
        metrics_data["Recall"] = [recall_train, recall_test]

    if "MSE" in metrics:
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        metrics_data["MSE"] = [train_mse, test_mse]

    evaluation = pd.DataFrame(metrics_data, index=["Train", "Test"])
    st.write(evaluation)


def report():
    st.title("Metrics Description")
    st.header('Precision:')
    st.write("Indicates the proportion of positive identifications (model predicted class 1) which were actually correct. A model which produces no false positives has a precision of 1.0.")
    st.header('Recall :')
    st.write("Indicates the proportion of actual positives which were correctly classified. A model which produces no false negatives has a recall of 1.0.")
    st.header('F1 score :')
    st.write("A combination of precision and recall. A perfect model achieves an F1 score of 1.0.")
    st.header('Support :')
    st.write("The number of samples each metric was calculated on.")
    st.header(' Accuracy :')
    st.write("The accuracy of the model in decimal form. Perfect accuracy is equal to 1.0")
    st.header('Macro avg :')
    st.write("Short for macro average, the average precision, recall and F1 score between classes. Macro avg doesn't class imbalance into effort, so if you do have class imbalances, pay attention to this metric.")
    st.header('Weighted avg :')
    st.write("Short for weighted average, the weighted average precision, recall and F1 score between classes. Weighted means each metric is calculated with respect to how many samples there are in each class. This metric will favour the majority class (e.g. will give a high vile when one class out performs another due to having more samples).")




def classification():
    with st.sidebar:
        st.title("Classification")
        model = st.selectbox("Model",["SVC","Tree","KNN","RandomFrorest","Voting","Bagging"])
        # col3,col4= st.columns(2)
        # gridsearch = col3.checkbox("GridSearch")
        # randomsearch = col4.checkbox("RandomSearch")
        with st.sidebar:
            
             option = st.radio("Navigation",["GridSearch","RandomSearch","Predict","Compare models"])

        # st.title("Prediction")
        # kind = st.checkbox("One Predict")


        
    df = pd.read_csv("dataset.csv")
    x = df.iloc[:,1:10]
    y= df.iloc[:,11]
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
    
  
        # if mesure:
        #     col1, col2, col3 = st.columns(3)
        #     col1.metric("Accurancy", round(svc.score(x,y),3))
        #     col2.metric("MSE", "9 mph")
        #     col3.metric("F1SCORE", "86%")
        
    

    if option=="GridSearch":

            if model == "SVC":
                global svc
                svc = svm.SVC(C= 0.1, gamma=1, kernel='linear')
                svc.fit(x_train, y_train)
                st.header("SVC Params")
                namec,firstc,midlec=st.columns(3)
                Min_value_C = firstc.number_input("Min_value C")
                Max_value_C = midlec.number_input("Max_value C")
                namec.header("C:")
                namegamma,firstgamma,midlegamma=st.columns(3)
                Min_value_g = firstgamma.number_input("Min_value Gamma")
                Max_value_g = midlegamma.number_input("Max_value Gamma")
                namegamma.header("GAMMA:")
                    
                name,last=st.columns(2)
                name.header("Kernel :")
                kernel = last.multiselect("Choose Kernel(s)",["sigmoid","poly","rbf"])
                #st.write(kernel)
                v1,v2 = st.columns(2)

                valider = v1.checkbox("Valider")
                
                save = v2.button("Save model")
                
                if valider:
                    params = {"C":np.arange(Min_value_C,Max_value_C,0.1),"gamma":np.arange(Min_value_g,Max_value_g,0.1),"kernel":kernel}
                    grid = GridSearchCV(svc,params)
                    grid.fit(x_train,y_train)
                    grid.best_params_
                    svc = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, svc.predict(x_test)))
                        report()
                        # col1, col2, col3 = st.columns(3)
                        
                        # col2.metric("MSE", "9 mph")
                        # col3.metric("F1SCORE", "86%")
                        # fig, ax = plt.subplots()
                        # fig=go.Figure(go.Indicator(mode="gauge+number+delta",value=round(svc.score(x,y),3),title={"text":"Accurancy"}))
                        # fig.update_layout()
                        # col1.write(fig)
                        # fig=go.Figure(go.Indicator(mode="gauge+number+delta",value=round(svc.score(x,y),3),title={"text":"Accurancy"}))
                        # fig.update_layout()
                        # col2.write(fig)
                    if result=="CURVE": 
                        
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix") 
                        fig, ax = plt.subplots()
                        cm=confusion_matrix(y_test,svc.predict(x_test),labels=svc.classes_)
                        st.write(cm)
                        ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=svc.classes_)
                        st.pyplot(fig)
                        st.subheader("ROC CURVE")
                        fig1, ax2 = plt.subplots()
                        ax = RocCurveDisplay(svc, x, y)
                        st.pyplot(fig1)

                if save:
                    filename = "svc"
                    pickle.dump(svc,open(filename,"wb"))
                    st.balloons()

                # with st.sidebar:
                    
            if model == "KNN":
                global mknn
                x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
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
                    grid.best_params_
                    mknn = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, mknn.predict(x_test)))
                        report()
                    if result=="CURVE": 
                        
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix") 
                        fig, ax = plt.subplots()
                        cm=confusion_matrix(y_test,mknn.predict(x_test),labels=mknn.classes_)
                        st.write(cm)
                        ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mknn.classes_)
                        st.pyplot(fig)
                if save:
                        filename = "mknn"
                        pickle.dump(mknn,open(filename,"wb"))
                        st.balloons()

            if model == "Tree":
                global mtree
                x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
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
                    grid.best_params_
                    mtree = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, mtree.predict(x_test)))
                        report()

                    if result=="CURVE": 
                        
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix") 
                        fig, ax = plt.subplots()
                        cm=confusion_matrix(y_test,mtree.predict(x_test),labels=mtree.classes_)
                        st.write(cm)
                        ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mtree.classes_)
                        st.pyplot(fig)
                        st.subheader("TREEPLOT")
                        apprent = mtree.fit(x_train,y_train)
                        fig, ax = plt.subplots()
                        tree.plot_tree(apprent, filled=True)
                        st.pyplot(fig)
                    if save:
                        filename = "mtree"
                        pickle.dump(mtree,open(filename,"wb"))
                        st.balloons()
            if model == "RandomFrorest":
                global raf
                x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
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
                    grid.best_params_
                    raf = grid.best_estimator_
                    result = st.selectbox("choose",["Metrics","CURVE"])
                    if result=="Metrics":
                        st.text('Model Report:\n ' + classification_report(y_test, raf.predict(x_test)))
                        report()

                    if result=="CURVE": 
                        
                        y_test = y_test.to_numpy()
                        st.subheader("Confusion Matrix") 
                        fig, ax = plt.subplots()
                        cm=confusion_matrix(y_test,raf.predict(x_test),labels=raf.classes_)
                        st.write(cm)
                        ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=raf.classes_)
                        st.pyplot(fig)
                        st.subheader("TREEPLOT")
                        apprent = raf.fit(x_train,y_train)
                        fig, ax = plt.subplots()
                        tree.plot_tree(apprent, filled=True)
                        st.pyplot(fig)
                    if save:
                        filename = "raf"
                        pickle.dump(raf,open(filename,"wb"))
                        st.balloons()
