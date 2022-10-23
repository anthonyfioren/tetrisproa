
import streamlit as st
st.header('ANALISIS KEGAGALAN EVOS LEGEND KE PLAYOFF MPL S10')
st.write('## Analisis Kegagalan Evos Legend melaju ke Playoff MPL Season 10 dilihat dari segi `Statistik` ')
st.write('Evos Legends gagal merebut tiket playoff MPL ID S10 setelah dikalahkan RRQ Hoshi. Sutjusin dan kawan-kawan dipaksa mengakui ketangguhan RRQ Hoshi dengan skor 0-2 pada laga kedua pekan kedelapan MPL ID S10, Sabtu (1/10).')
st.write('Mari kita bahas kegagalan ini jika dilihat darii segi statistik')
st.markdown("# Kilas Balik")
#Load packages

import numpy as np

import pandas as pd

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import seaborn as sns
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
st.write('Pengecekan statistik dilakukan dengan analisis korelasi antara statistik di dalam permainan Mobile Legend di Turnament MPL ID S10')

plt.figure(figsize = (24, 14))
corr = df.corr()
heattt= sns.heatmap(corr, annot = True, linewidths = 1)
plt.show()
st.pyplot(plt.gcf())
st.write('Dari peta korelasi heatmap terlihat bahwa total kill, gold, damage, dan tortoise kill sangat berpengaruh. Untuk selanjutnya kita akan tengok plot barnya')




# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(2, 2)
  
# For Sine Function
axis[0, 0].bar(df.Nama_Tim, df.Kills,  color=["#d0d0d0","#d0d0d0","#d0d0d0",'blue',"#d0d0d0","#d0d0d0","#d0d0d0","#d0d0d0"], width=0.6, edgecolor='black')
axis[0, 0].set_title(f'Total Kill(s)', fontweight='bold')

# For Cosine Function
axis[0, 1].bar(df.Nama_Tim, df.Gold,  color=["#d0d0d0","#d0d0d0","#d0d0d0",'blue',"#d0d0d0","#d0d0d0","#d0d0d0","#d0d0d0"], width=0.6, edgecolor='black')
axis[0, 1].set_title(f'Total Gold(s)', fontweight='bold')

# For Tangent Function
axis[1, 0].bar(df.Nama_Tim, df.Damage,  color=["#d0d0d0","#d0d0d0","#d0d0d0",'blue',"#d0d0d0","#d0d0d0","#d0d0d0","#d0d0d0"], width=0.6, edgecolor='black')
axis[1, 0].set_title(f'Total Damage(s)', fontweight='bold')

# For Tanh Function
axis[1, 1].bar(df.Nama_Tim, df.Tortoise_Kills,  color=["#d0d0d0","#d0d0d0","#d0d0d0",'blue',"#d0d0d0","#d0d0d0","#d0d0d0","#d0d0d0"], width=0.6, edgecolor='black')
axis[1, 1].set_title(f'Total Tortoise Kill(s)', fontweight='bold')
plt.show()
st.pyplot(plt.gcf())

st.write('Mari kita lihat dari grafik lain yang dapat ditampilkan sebagai berikut')

plt.figure(figsize = (24, 14))
sns.relplot(
    data=df, x="Kills", y="Assist", size="Win", sizes=(15, 200)
)
plt.title('Hubungan antar Jumlah Kill Turtle, Lord Dengan kemenangan', fontsize=10, color = 'blue')
plt.show()
st.pyplot(plt.gcf())

sns.relplot(
    data=df, x="Tortoise_Kills", y="Lord_Kills", size="Win", sizes=(15, 200)
)
plt.title('Hubungan antar Jumlah Kill Turtle, Lord Dengan kemenangan', fontsize=10, color = 'blue')
plt.ylabel('Turtle')
plt.xlabel('Lord')

st.write('Terlihat Evos (dengan 15 win) tertinggal diantara yang lain')