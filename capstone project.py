
import streamlit as st
st.header('ANALISIS KEGAGALAN EVOS LEGEND KE PLAYOFF MPL S10')
st.write('## Analisis Kegagalan Evos Legend melaju ke Playoff MPL Season 10 dilihat dari segi `Statistik` ')
st.write('Evos Legends gagal merebut tiket playoff MPL ID S10 setelah dikalahkan RRQ Hoshi. Sutjusin dan kawan-kawan dipaksa mengakui ketangguhan RRQ Hoshi dengan skor 0-2 pada laga kedua pekan kedelapan MPL ID S10, Sabtu (1/10).')
st.write('Mari kita bahas kegagalan ini jika dilihat darii segi statistik')
st.markdown("# Kilas Balik")
#Load packages

import numpy as np
import seaborn as sns
import pandas as pd
import datetime as dt
import time
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score

import plotly
import plotly.graph_objects as go
import plotly.express as px

#request data from google drive

url = 'https://drive.google.com/file/d/1brHF2q3srKp7hy9vGigpCBK8TOnmbHmP/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)

season=[1,2,3,4,5,6,7,8,9,10]
klasemen=[2,4,6,1,3,6,2,4,4,7]
Playoff=[6,6,6,6,6,6,6,6,6,6]
radius = 15

plt.plot(season, klasemen, color='blue', marker='o')
plt.title('KLASEMEN EVOS ALL TIME', fontsize=14)
plt.xlabel('Season', fontsize=14)
plt.ylabel('Peringkat', fontsize=14)
plt.ylim(8, 1)
plt.grid(True)
plt.plot(season[9],klasemen[9], 'o',
   ms=radius * 2, mec='red', mfc='none', mew=2)
plt.plot(season, Playoff, color='red') 
plt.show()
st.pyplot(plt.gcf())
st.write('Grafik yang tersedia menunjukan bahwa performa evos musim ini sangatlah buruk bila dibanding musim sebelumnya, bahkan mereka gagal merebut tiket playoff setelah finish di peringkat 7')

st.markdown("# Statistik")


plt.figure(figsize = (24, 14))
corr = df.corr()
heattt= sns.heatmap(corr, annot = True, linewidths = 1)
plt.show()
st.pyplot(plt.gcf())
