import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title='K-Means Clustering', layout="wide", initial_sidebar_state='expanded')

st.write("""# Agrupamento com K-Means

Esta aplicação usa K-Means para segmentar clientes com base em seu comportamento de compra, considerando:
- Recência (R): Quantidade de dias desde a última compra.
- Frequência (F): Quantidade total de compras no período.
- Valor (V): Total de dinheiro gasto nas compras do período.

O objetivo é agrupar clientes com padrões semelhantes para melhor direcionamento de ações de marketing e CRM.
""")
st.markdown("---")

@st.cache_data
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def nomear_clusters(modelo, df):
    nomes_clusters = {
        0: "Clientes VIP",
        1: "Clientes Frequentes",
        2: "Clientes Ocasionalmente Ativos",
        3: "Clientes Inativos",
        4: "Clientes Novos",
        5: "Clientes Potenciais",
        6: "Clientes de Baixo Gasto",
        7: "Clientes de Alto Valor",
        8: "Clientes de Risco",
        9: "Clientes Regulares"
    }
    df['Cluster_Nome'] = df['Cluster'].map(nomes_clusters)
    return df

def main():
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Dados de compras", type=['csv', 'xlsx'])
    
    if data_file_1 is not None:
        df_compras = pd.read_csv(data_file_1, infer_datetime_format=True, parse_dates=['DiaCompra'])
        dia_atual = df_compras['DiaCompra'].max()
        
        st.write('## Processando os dados')
        
        df_recencia = df_compras.groupby(by='ID_cliente', as_index=False)['DiaCompra'].max()
        df_recencia.columns = ['ID_cliente', 'DiaUltimaCompra']
        df_recencia['Recencia'] = df_recencia['DiaUltimaCompra'].apply(lambda x: (dia_atual - x).days)
        df_recencia.drop('DiaUltimaCompra', axis=1, inplace=True)
        
        df_frequencia = df_compras.groupby('ID_cliente')['CodigoCompra'].count().reset_index()
        df_frequencia.columns = ['ID_cliente', 'Frequencia']
        
        df_valor = df_compras.groupby('ID_cliente')['ValorTotal'].sum().reset_index()
        df_valor.columns = ['ID_cliente', 'Valor']
        
        df_RFV = df_recencia.merge(df_frequencia, on='ID_cliente').merge(df_valor, on='ID_cliente')
        df_RFV.set_index('ID_cliente', inplace=True)
        
        st.write('## Aplicando K-Means')
        k = st.sidebar.slider('Escolha o número de clusters', min_value=2, max_value=10, value=4)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df_RFV['Cluster'] = kmeans.fit_predict(df_RFV[['Recencia', 'Frequencia', 'Valor']])
        
        df_RFV = nomear_clusters(kmeans, df_RFV)
        
        st.write('Distribuição dos clusters:')
        st.write(df_RFV[['Cluster', 'Cluster_Nome']].value_counts())
        
        df_xlsx = to_excel(df_RFV)
        st.download_button(label='📥 Download dos resultados', data=df_xlsx, file_name='KMeans_Resultados.xlsx')

if __name__ == '__main__':
    main()
