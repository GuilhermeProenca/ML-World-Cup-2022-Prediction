import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import joblib

image = Image.open("figs/fifa-world-cup-catar-2022-logo.jpg")
st.image(image)

st.title("World Cup Qatar 2022 match prediction")

df_selecoes = pd.read_csv("datasets/Selecoes2022.csv")

todas_selecoes = sorted(df_selecoes['Selecoes'].unique())

selecionar_primeira_selecao = st.selectbox('First team (playing as home team)', todas_selecoes)

selecao_b = df_selecoes[df_selecoes['Selecoes'] != selecionar_primeira_selecao]
selecao_b_sorted = sorted(selecao_b['Selecoes'].unique())
selecionar_segunda_selecao = st.selectbox('Second team (playing as a visiting team)', selecao_b_sorted)

model = joblib.load('model.pkl')

nome_time = {'France': 0, 'Mexico': 1, 'Chile': 2, 'Romania': 3, 'Peru': 4, 'Argentina': 5, 'Yugoslavia': 6, 'Bolivia': 7, 'Switzerland': 8, 'Netherlands': 9,
            'Germany': 10, 'Belgium': 11, 'Czechoslovakia': 12, 'Italy': 13, 'Spain': 14, 'Austria': 15, 'Hungary': 16, 'Dutch East Indies': 17, 'Sweden': 18,
            'Cuba': 19, 'Brazil': 20, 'USA': 21, 'England': 22, 'Uruguay': 23, 'Scotland': 24, 'Turkey': 25, 'Korea Republic': 26, 'Northern Ireland': 27,
            'Soviet Union': 28, 'Colombia': 29, 'Bulgaria': 30, 'Korea DPR': 31, 'Portugal': 32, 'El Salvador': 33, 'Australia': 34, 'Wales': 35, 'Paraguay': 36,
            'Morocco': 37, 'Zaire': 38, 'Poland': 39, 'Haiti': 40, 'Algeria': 41, 'Cameroon': 42, 'Iraq': 43, 'Canada': 44, 'Egypt': 45, 'rn">Republic of Ireland': 46,
            'Norway': 47, 'Saudi Arabia': 48, 'Greece': 49, 'Russia': 50, 'South Africa': 51, 'Croatia': 52, 'Nigeria': 53, 'Slovenia': 54, 'IR Iran': 55, 'Tunisia': 56,
            'New Zealand': 57, 'Denmark': 58, 'Costa Rica': 59, 'rn">United Arab Emirates': 60, "Côte d'Ivoire": 61, 'Iran': 62, 'rn">Serbia and Montenegro': 63, 'Ukraine': 64,
            'Ecuador': 65, 'Ghana': 66, 'rn">Bosnia and Herzegovina': 67, 'Panama': 68, 'Japan': 69, 'Jamaica': 70, 'Senegal': 71, 'China PR': 72, 'Togo': 73, 'Angola': 74,
            'rn">Trinidad and Tobago': 75, 'Serbia': 76, 'Slovakia': 77, 'Honduras': 78, 'Iceland': 79, 'Israel': 80, 'Kuwait': 81, 'Czech Republic': 82, 'Iran ': 83, 'Qatar': 84,
            'South Korea': 85, 'United States': 86}

df_campeoes = pd.read_csv("datasets/Campeoes.csv")
campeoes = df_campeoes['Vencedor'].value_counts()

def predicao(timeA, timeB):
  idA = nome_time[timeA]
  idB = nome_time[timeB]
  campeaoA = campeoes.get(timeA) if campeoes.get(timeA) != None else 0
  campeaoB = campeoes.get(timeB) if campeoes.get(timeB) != None else 0

  x = np.array([idA, idB, campeaoA, campeaoB]).astype('float64')
  x = np.reshape(x, (1,-1))
  _y = model.predict_proba(x)[0]

  text = (timeA+"'s chance to win: {}%\n\n"+timeB+"'s chance to win: {}%\n\nDraw chance: {}%").format(round(_y[1]*100, 2), 
                                                                                                    round(_y[2]*100, 2), 
                                                                                                    round(_y[0]*100, 2))
  return _y[0], text

prob1, text1 = predicao(selecionar_primeira_selecao, selecionar_segunda_selecao)

if st.button('Prediction'):
    st.text(text1)