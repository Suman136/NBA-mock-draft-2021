import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('p-t-17_brojevi.csv')
data2= pd.read_csv('p-t-18_brojevi.csv')
data3= pd.read_csv('p-t-19_brojevi.csv')
data4= pd.read_csv('p-t-20_brojevi.csv')
#data5= pd.read_csv('p-t-18_brojevi.csv')

data=data[['Team',  'FGA_T',
       'FG%_T', '3PA_T', '3P%_T',  'FTA_T', 'FT%_T', 'TOV_T',
       'ORB_T', 'DRB_T', 'RPG_T', 'APG_T', 'SPG_T', 'BPG_T', 'PPG_T',
       'TS%_T', 'eFG%_T', 'Total S%_T', 'ORB%_T', 'DRB%_T', 'TRB%_T', 'AST%_T',
       'TOV%_T', 'STL%_T', 'BLK%_T', 'PPS_T', 'FIC40_T', 'ORtg_T', 'DRtg_T',
       'eDiff_T', 'Poss_T', 'Pace_T', 'FG', 'FGA', 'FG%', '2P', '2P%',
       '3P',  '3P%', 'FT', 'FT%', 'ORB', 'DRB', 'TRB', 'AST',
       'STL', 'BLK', 'TOV', 'PTS', 'PER', 'TS%', 'eFG%', '3PAr', 'FTr',
       'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS',
       'DWS', 'WS', 'WS/40', 'OBPM', 'DBPM', 'BPM' , 'Age', 'Player','ID']]

data2=data2[['Team',  'FGA_T',
       'FG%_T', '3PA_T', '3P%_T',  'FTA_T', 'FT%_T', 'TOV_T',
       'ORB_T', 'DRB_T', 'RPG_T', 'APG_T', 'SPG_T', 'BPG_T', 'PPG_T',
       'TS%_T', 'eFG%_T', 'Total S%_T', 'ORB%_T', 'DRB%_T', 'TRB%_T', 'AST%_T',
       'TOV%_T', 'STL%_T', 'BLK%_T', 'PPS_T', 'FIC40_T', 'ORtg_T', 'DRtg_T',
       'eDiff_T', 'Poss_T', 'Pace_T', 'FG', 'FGA', 'FG%', '2P', '2P%',
       '3P',  '3P%', 'FT', 'FT%', 'ORB', 'DRB', 'TRB', 'AST',
       'STL', 'BLK', 'TOV', 'PTS', 'PER', 'TS%', 'eFG%', '3PAr', 'FTr',
       'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS',
       'DWS', 'WS', 'WS/40', 'OBPM', 'DBPM', 'BPM' , 'Age', 'Player','ID']]

data3=data3[['Team',  'FGA_T',
       'FG%_T', '3PA_T', '3P%_T',  'FTA_T', 'FT%_T', 'TOV_T',
       'ORB_T', 'DRB_T', 'RPG_T', 'APG_T', 'SPG_T', 'BPG_T', 'PPG_T',
       'TS%_T', 'eFG%_T', 'Total S%_T', 'ORB%_T', 'DRB%_T', 'TRB%_T', 'AST%_T',
       'TOV%_T', 'STL%_T', 'BLK%_T', 'PPS_T', 'FIC40_T', 'ORtg_T', 'DRtg_T',
       'eDiff_T', 'Poss_T', 'Pace_T', 'FG', 'FGA', 'FG%', '2P', '2P%',
       '3P',  '3P%', 'FT', 'FT%', 'ORB', 'DRB', 'TRB', 'AST',
       'STL', 'BLK', 'TOV', 'PTS', 'PER', 'TS%', 'eFG%', '3PAr', 'FTr',
       'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS',
       'DWS', 'WS', 'WS/40', 'OBPM', 'DBPM', 'BPM' , 'Age', 'Player','ID']]

data4=data4[['Team',  'FGA_T',
       'FG%_T', '3PA_T', '3P%_T',  'FTA_T', 'FT%_T', 'TOV_T',
       'ORB_T', 'DRB_T', 'RPG_T', 'APG_T', 'SPG_T', 'BPG_T', 'PPG_T',
       'TS%_T', 'eFG%_T', 'Total S%_T', 'ORB%_T', 'DRB%_T', 'TRB%_T', 'AST%_T',
       'TOV%_T', 'STL%_T', 'BLK%_T', 'PPS_T', 'FIC40_T', 'ORtg_T', 'DRtg_T',
       'eDiff_T', 'Poss_T', 'Pace_T', 'FG', 'FGA', 'FG%', '2P', '2P%',
       '3P',  '3P%', 'FT', 'FT%', 'ORB', 'DRB', 'TRB', 'AST',
       'STL', 'BLK', 'TOV', 'PTS', 'PER', 'TS%', 'eFG%', '3PAr', 'FTr',
       'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS',
       'DWS', 'WS', 'WS/40', 'OBPM', 'DBPM', 'BPM' , 'Age', 'Player','ID']]


del data['Player']
del data2['Player']
del data3['Player']
del data4['Player']
#del data5['Player']

X1 = data.iloc[:,1:-1].values
Y1 = data.iloc[:,-1].values


X1=np.asarray(X1,float)
sc = StandardScaler()
X1 = sc.fit_transform(X1)


X2 = data2.iloc[:,1:-1].values
Y2 = data2.iloc[:,-1].values


X2=np.asarray(X2,float)
X2 = sc.fit_transform(X2)


X3 = data3.iloc[:,1:-1].values
Y3 = data3.iloc[:,-1].values


X3=np.asarray(X3,float)
X3 = sc.fit_transform(X3)


X4 = data4.iloc[:,1:-1].values
Y4 = data4.iloc[:,-1].values


X4=np.asarray(X4,float)
X4 = sc.fit_transform(X4)


#X5 = data5.iloc[:,3:-2].values
#Y5 = data5.iloc[:,-1].values


#X5=np.asarray(X5,float)
#X5 = sc.fit_transform(X5)

array_tuple = (X1, X2, X3, X4)
X = np.vstack(array_tuple)


Y=np.append(Y1,Y2,0)
Y=np.append(Y,Y3,0)
Y=np.append(Y,Y4,0)
#Y=np.append(Y,Y5,0)

ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=1000, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=1000, activation='relu'))

# Adding the third hidden layer

ann.add(tf.keras.layers.Dense(units=1000, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1000, activation='relu'))



# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
#ann.compile(optimizer = 'Adadelta', loss = tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
ann.compile(optimizer = 'adam', loss = tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])


# Training the ANN on the Training set
ann.fit(X, Y, batch_size = 100, epochs = 1000)

data2021=pd.read_csv('stats20212.csv')
raspored=pd.read_csv('raspored-2021.csv')
tiiim=raspored.iloc[:,:].values



data2021=data2021[['Team',  'FGA_T',
       'FG%_T', '3PA_T', '3P%_T',  'FTA_T', 'FT%_T', 'TOV_T',
       'ORB_T', 'DRB_T', 'RPG_T', 'APG_T', 'SPG_T', 'BPG_T', 'PPG_T',
       'TS%_T', 'eFG%_T', 'Total S%_T', 'ORB%_T', 'DRB%_T', 'TRB%_T', 'AST%_T',
       'TOV%_T', 'STL%_T', 'BLK%_T', 'PPS_T', 'FIC40_T', 'ORtg_T', 'DRtg_T',
       'eDiff_T', 'Poss_T', 'Pace_T', 'FG', 'FGA', 'FG%', '2P', '2P%',
       '3P',  '3P%', 'FT', 'FT%', 'ORB', 'DRB', 'TRB', 'AST',
       'STL', 'BLK', 'TOV', 'PTS', 'PER', 'TS%', 'eFG%', '3PAr', 'FTr',
       'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS',
       'DWS', 'WS', 'WS/40', 'OBPM', 'DBPM', 'BPM' , 'Age', 'Player']]

draft_lista=['NBA draft']
postotak=['Percentage']

for i in range(len(tiiim)):
  team=tiiim[i,1]
  draft_lista.append(team)
  tablica=[]

  data2021=data2021[data2021.Player != str(draft_lista[-2])]

  filter=data2021[data2021.Team==team]
  K= filter.iloc[:,1:-1].values
  T=filter.iloc[:,-1].values
  

  C=np.asarray(K,float)
  C = sc.fit_transform(C)

  for a in range(len(filter)):

    prvi=ann.predict([[C[a,0],C[a,1],C[a,2],C[a,3],C[a,4],C[a,5],C[a,6],C[a,7],C[a,8],
                       C[a,9],C[a,10],C[a,11],C[a,12],C[a,13],C[a,14],C[a,15],C[a,16],
                       C[a,17],C[a,18],C[a,19],C[a,20],C[a,21],C[a,22],C[a,23],C[a,24],
                       C[a,25],C[a,26],C[a,27],C[a,28],C[a,29],C[a,30],C[a,31],C[a,32],
                       C[a,33],C[a,34],C[a,35],C[a,36],C[a,37],C[a,38],C[a,39],C[a,40],
                       C[a,41],C[a,42],C[a,43],C[a,44],C[a,45],C[a,46],C[a,47],C[a,48],
                       C[a,49],C[a,50],C[a,51],C[a,52],C[a,53],C[a,54],C[a,55],C[a,56],
                       C[a,57],C[a,58],C[a,59],C[a,60],C[a,61],C[a,62],C[a,63],C[a,64],
                       C[a,65],C[a,66],C[a,67],C[a,68]]])
                       
    tablica.append(prvi)
    print(tablica)
  choosen = max(tablica)
  print (choosen)
  postotak.append(choosen)
  for b in range(len(filter)):
    if choosen==tablica[b]:

      odabrani_igrac= T[b]
      draft_lista.append(odabrani_igrac)
      print(odabrani_igrac)

choosen_draft = pd.DataFrame(draft_lista[1::2], columns=['Team'])
choosen_draft['Player']=draft_lista[2::2]
choosen_draft['Percentage']=postotak[1:]
print(choosen_draft)
