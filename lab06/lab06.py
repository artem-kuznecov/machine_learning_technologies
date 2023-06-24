import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import plotly.express as px

X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)


# Чтобы в тесте получилось низкое качество используем только 0,5% данных для обучения
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=1)

# Модели
models_list = ['LogR', 'KNN_5', 'Tree', 'RF', 'GB']
clas_models = {'LogR': LogisticRegression(), 
               'KNN_5':KNeighborsClassifier(n_neighbors=5),
               'Tree':DecisionTreeClassifier(),
               'RF':RandomForestClassifier(),
               'GB':GradientBoostingClassifier()}



def plt_plot():
    # create a new plot

    plt.scatter(X_test[:50, 0], X_test[:50, 1],
            color='blue', marker='o')
    plt.scatter(X_test[50:100, 0], X_test[50:100, 1],
            color='green', marker='s')
 
    plt.xlabel('1')
    plt.ylabel('2')
    plt.legend(loc='upper left')

    return plt


def run_models(models_input):
    roc_auc_dict = {}
    for model_name in models_input:
        model = clas_models[model_name]
        model.fit(X_train, y_train)
        # Предсказание значений
        Y_pred = model.predict(X_test)
        # Предсказание вероятности класса "1" для roc auc
        Y_pred_proba_temp = model.predict_proba(X_test)
        Y_pred_proba = Y_pred_proba_temp[:,1]
        roc_auc = roc_auc_score(y_test, Y_pred_proba)
        roc_auc_dict[model_name] = roc_auc
    return roc_auc_dict



#Входные компоненты
models_input = gr.inputs.CheckboxGroup(models_list, type='value', label='Выберите модели')

#Выходные компоненты
out_label = gr.outputs.Label(type='confidences', label='ROC AUC')
outputs = gr.Plot()

iface = gr.Interface(
  fn=run_models,
  inputs=[models_input], 
  outputs=[out_label],
  title='Модели машинного обучения')

plot = gr.Interface(fn=plt_plot, inputs=None, outputs=outputs)

run = gr.TabbedInterface([iface, plot])
run.launch()