import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

matplotlib.use('Agg')
plt.style.use('ggplot')

def treina_modelo(model, test_size, variavel_y, data):
    print('treinou o modelo')
    le = LabelEncoder()
    for col in data:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col])
            print("foi")
        else:
            #print(col)
            print("foi nada")
    y = data[variavel_y]
    x = data.drop('Exited', axis=1)

    test_size = float(test_size)

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=test_size)

    model.fit(x_treino, y_treino)
    preds = model.predict(x_teste)
    acertos = accuracy_score(preds,  y_teste)
    print('treinou o modelo2')
    print(acertos)
    return acertos

def main():
    st.sidebar.title('Sobre o applicativo:')
    st.sidebar.markdown('desenvolvi esse aplicativo para analisar/testar a base de dados, durante'
                        ' a semana de aceleracao data-sciensce da codenation')
    st.sidebar.markdown('meu Linkedin https://www.linkedin.com/in/carlos-alberto-ochner-412946192/')
    st.sidebar.markdown('Metrica usada para classificao: Accuracy_score do sklearn')
    st.sidebar.markdown('Metrica usada para Regressao: MAE(mean_absolute_error) tambem do sklearn')
    #['Site da Codenation'](upload:https://www.linkedin.com/in/carlos-alberto-ochner-412946192/)
    st.title("Database analysis")
    #st.image('https://media.giphy.com/media/KyBX9ektgXWve/giphy.gif', width=200)
    #st.header("Lets do some cool ml stuf guys")
    file = st.file_uploader('Escolha A base de dados que deseja analisar', type='csv')

    if file is not None:
        st.subheader('Analise Dos Dados')
        data = pd.read_csv(file)
        st.markdown('**Shape dos dados:**')
        st.text('Linhas {}, Colunas {} '.format(data.shape[0], data.shape[1]))
        numero = st.slider('Quantas colunas deseja ver dos dados ?', min_value=3, max_value=20)
        st.dataframe(data.head(numero))
        st.markdown('**Nome Das Colunas:**')
        st.markdown(list(data.columns))
        coisas = pd.DataFrame({'nomes': data.columns, 'tipos': data.dtypes, 'faltantes_totais':data.isna().sum(), 'faltantes_porcem': (data.isna().sum() / data.shape[0]) * 100})
        st.markdown('**Colunas do tipo int:**')
        st.markdown(list(coisas[coisas['tipos'] == 'int64']['nomes']))
        st.markdown('**Colunas do tipo Float:**')
        st.markdown(list(coisas[coisas['tipos'] == 'float64']['nomes']))
        st.markdown('**Colunas do tipo Object: **')
        st.markdown(list(coisas[coisas['tipos'] == 'object']['nomes']))

        # VALORES NULLOS
        if st.checkbox('Deseja ver a analise dos Valores Nullos?'):
            st.table(coisas[coisas['faltantes_totais']!= 0][['faltantes_totais', 'faltantes_porcem']])
            #Filtro para selecionar valores nulos
            valores_100 = coisas[coisas['faltantes_porcem']== 100]['nomes']
            contagem = len(valores_100)
            valores_80 = coisas[coisas['faltantes_porcem'] >= 80]['nomes']
            valores_50 = coisas[coisas['faltantes_porcem'] >= 50]['nomes']
            valores_30 = coisas[coisas['faltantes_porcem'] >= 30]['nomes']
            valores_15 = coisas[coisas['faltantes_porcem'] >= 15]['nomes']

            st.markdown('total de colunas 100% nulas: {}'.format(contagem))

            st.markdown('total de colunas 80% ou mais nulas: {}'.format(len(valores_80)))

            st.markdown('total de colunas 50% ou mais nulas: {}'.format(len(valores_50)))

            st.markdown('total de colunas 30% ou mais nulas: {}'.format(len(valores_30)))

            st.markdown('total de colunas 15% ou mais nulas: {}'.format(len(valores_15)))

## # # ## # ### # # #PRENCHEMENTO DE DADOS NULLOS # # # ## ## # #

            if st.checkbox("Tratamento de Valores Nullos"):
               tratamento  = st.selectbox("Metodos para Prencher", ['Media', 'Mediana'])
               if st.button('Tratar Nan'):
                   if tratamento == 'Media':
                       for col in data:
                           if data[col].isna().sum() > 0:
                               media_coluna = data[col].mean()
                               data[col] = data[col].fillna(media_coluna, inplace=True)
                   if tratamento == 'Mediana':
                       for col in data:
                            if data[col].isna().sum() > 0:
                               mediana_coluna = data[col].median()
                               data[col] = data[col].fillna(mediana_coluna)
                   st.success('Valores prenchidos com a {}'.format(tratamento))
        # #  # # ## # # ## # # ## ## # # # # ## ## # # ## # #
        if st.checkbox('Deseja ver o Summario dos dados ?'):
            st.write(data.describe())

        st.title('Analise Visual dos dados')
        # Correlation plot

        #Seaborn Plot
        if st.checkbox('Correlation Map'):
            st.write(sns.heatmap(data.corr(), annot=True))
            st.pyplot()
        #Count Plot
        if st.checkbox('Grafico de contagem'):
            all_columns_names = data.columns.tolist()
            primary_col = st.selectbox('Qual a coluna primaria para ser analisada?', all_columns_names)
            selected_columns_names = st.multiselect('Colunas selecionadas', all_columns_names)
            #st.text('Grafico Gerado')
            if selected_columns_names:
                vc_plot = data.groupby(primary_col)[selected_columns_names].count()
            else:
                vc_plot = data.iloc[:, -1].value_counts()
            st.write(vc_plot.plot(kind='bar'))
            st.pyplot()

        #Pie Chart
        if st.checkbox('Pie Plot'):
            all_columns_names = data.columns.tolist()
            if st.button('Gerar o grafico'):
                st.success('grafico gerado com sucesso')
                st.write(data.iloc[:, -1].value_counts().plot.pie(autopct='%1.1f%%'))
                st.pyplot()



        # Grafico customizavel
        st.subheader('Grafcico Customizavel')
        all_columns_names = data.columns.tolist()
        type_of_plots = st.selectbox("Selecione o tipo do grafico", ['area', 'bar', 'line', 'hist', 'box', 'kde'])
        selected_columns_names = st.multiselect("Selecione as colunas para plotagem", all_columns_names)

        if st.button("Gerar grafico"):
            st.success("Gerando um grafico de {} para {}".format(type_of_plots, selected_columns_names))
            #Plotando o graficoooooooo
            if type_of_plots == 'area':
                cust_data = data[selected_columns_names]
                st.area_chart(cust_data)
            elif type_of_plots == 'bar':
                cust_data = data[selected_columns_names]
                st.bar_chart(cust_data)
            elif type_of_plots == 'line':
                cust_data = data[selected_columns_names]
                st.line_chart(cust_data)

            # Custom plot usando matplotlib
            elif type_of_plots:
                cust_plot = data[selected_columns_names].plot(kind=type_of_plots)
                st.write(cust_plot)
                st.pyplot();

        if st.checkbox("Testar a base de dados"):
            le = LabelEncoder()
            if st.checkbox("Algoritimos de classificacao"):
                modelo_selcionado = st.selectbox("Escolha um modelo", ['DummyClassifier', 'KnnClassifier', 'DecisionTree'])
                test_size = st.text_input('qual o tamanho do teste ?', '0.2')
                variavel_y = st.text_input("Qual a variavel que voce deseja prever ?")

                # SELECAO DE MODELOS PARA TREINAR - CLASSIFICACAO

                if st.button('Testar o Modelo'):
                    if modelo_selcionado == 'DummyClassifier':
                        model = DummyClassifier()
                        #treina_modelo(model_dummy, test_size, variavel_y, data)
                        for col in data:
                            if data[col].dtype == 'object':
                                data[col] = le.fit_transform(data[col])
                        y = data[variavel_y]
                        x = data.drop(variavel_y, axis=1)

                        test_size = float(test_size)

                        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=test_size)

                        model.fit(x_treino, y_treino)
                        preds = model.predict(x_teste)
                        acertos = accuracy_score(preds,  y_teste)
                        st.markdown('**A accuracia foi de: {}**'.format(acertos*100))
# # # # # # # ## # ## # # ## # # # # # # # KNN CLASSIFIER # # #### # # # # ## #@ # # ##
                    elif modelo_selcionado == 'KnnClassifier':
                        model_knn = KNeighborsClassifier()
                        #treina_modelo(model_knn, test_size, variavel_y, data)
                        for col in data:
                            if data[col].dtype == 'object':
                                data[col] = le.fit_transform(data[col])
                        y = data[variavel_y]
                        x = data.drop(variavel_y, axis=1)

                        test_size = float(test_size)

                        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=test_size)

                        model_knn.fit(x_treino, y_treino)
                        preds_knn = model_knn.predict(x_teste)
                        acertos_knn = accuracy_score(preds_knn,  y_teste)
                        st.markdown('**A accuracia foi de: {}**'.format(acertos_knn*100))
# # # # # # # # # ARVORE DE DECISAO # # # # # # ## # # # ## # # # #
                    elif modelo_selcionado == 'DecisionTree':
                        model_Tree = DecisionTreeClassifier()
                        for col in data:
                            if data[col].dtype == 'object':
                                data[col] = le.fit_transform(data[col])
                        y = data[variavel_y]
                        x = data.drop(variavel_y, axis=1)

                        test_size = float(test_size)

                        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=test_size)

                        model_Tree.fit(x_treino, y_treino)
                        preds_tree = model_Tree.predict(x_teste)
                        acertos_tree = accuracy_score(preds_tree,  y_teste)
                        st.markdown('**A accuracia foi de: {}**'.format(acertos_tree*100))

            if st.checkbox("Algoritimos de Regressao"):
                modelo_selcionado = st.selectbox("Escolher um modelo", ['DummyRegressor', 'KnnRegressor', 'DecisionTreeRegressor'])
                test_size = st.text_input('Qual o tamanho do teste?', '0.2')
                variavel_y = st.text_input("Qual a variavel que voce deseja prever?")
                if st.button('Testar o modelo'):
                    if modelo_selcionado == 'DummyRegressor':
                        model_dummy = DummyRegressor()
                        for col in data:
                            if data[col].dtype == 'object':
                                data[col] = le.fit_transform(data[col])

                        y = data[variavel_y]
                        x = data.drop(variavel_y, axis=1)

                        test_size = float(test_size)

                        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=test_size)

                        model_dummy.fit(x_treino, y_treino)
                        preds_dummy = model_dummy.predict(x_teste)
                        error = mean_absolute_error(y_teste, preds_dummy)
                        st.markdown('**A MAE foi de {}**'.format(error))
                    if modelo_selcionado == 'KnnRegressor':
                        model_knn_r = KNeighborsRegressor()
                        for col in data:
                            if data[col].dtype == 'object':
                                data[col] = le.fit_transform(data[col])
                        y = data[variavel_y]
                        x = data.drop(variavel_y, axis=1)

                        test_size = float(test_size)

                        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=test_size)

                        model_knn_r.fit(x_treino, y_treino)
                        preds_knn_r = model_knn_r.predict(x_teste)
                        error = mean_absolute_error(y_teste, preds_knn_r)
                        st.markdown('**A MAE foi de {}**'.format(error))

                    if modelo_selcionado == 'DecisionTreeRegressor':
                        model_tree_r = DecisionTreeRegressor()
                        for col in data:
                            if data[col].dtype == 'object':
                                data[col] = le.fit_transform(data[col])
                        y = data[variavel_y]
                        x = data.drop(variavel_y, axis=1)

                        test_size = float(test_size)

                        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=test_size)

                        model_tree_r.fit(x_treino, y_treino)
                        preds = model_tree_r.predict(x_teste)
                        error = mean_absolute_error(y_teste, preds)
                        st.markdown('**A MAE foi de {}**'.format(error))




if __name__ == '__main__':
    main()