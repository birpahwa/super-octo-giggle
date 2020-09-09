
#Preprocessing: data in single sheet:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df =pd.read_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands v2.xls',sheet_name='Sheet5')

df['year']= df.year.map(lambda x: x+2000).astype('Int32')

df1= df.reindex(columns=['Country Name','country code','year','pop','areakm2','gdpnom','hotrooms', 'expend',
'receipt','ovnarriv','dayvisit','crusvis','arrpleas','arrbus','arrair','arrwat','arram','arreap','arreur','arroth','arrausl','arrger','tcov','intxexg','intxexs','intxexal','intxcac','oteximg','otxims','otximal','otxcad'])
df2=df1[np.logical_or(df1.year>2006,df1.year==2007)]
df3=df2[df2.year!=2012]
df3 = df3.rename(columns={'country code': 'countryCode','Country Name':'CountryName','flights - WB':'flights-WB'})
df4 = df3[(df3['countryCode']!=9) & (df3['countryCode']!=11)& (df3['countryCode']!=14)& (df3['countryCode']!=15)& (df3['countryCode']!=20)& (df3['countryCode']!=23)& (df3['countryCode']!=24)]
df.to_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands v2.xlsx',sheet_name='Sheet5')

#Preprocessing to decide totals rows and attributes

df = pd.read_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands vvvv.xlsx')
count = df.isna().sum()
print(count)
df1=df.drop(['expend','hotrooms','crusvis', 'arreap','arroth', 'arrausl'
], axis=1)

print(df1)
count = df1.isna().sum()
print(count)
df2 = df1[(df1['countryCode']!=8)]
df2.to_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx',sheet_name='dataFrame')
df2['gdpnom'] = np.log(df2['gdpnom'])
df2['receipt'] = np.log(df2['receipt'])
df2.to_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx',sheet_name='dataFrame')
df.to_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx',sheet_name='Sheet5')




#Data Quality (imputing missing values)


df =pd.read_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx')
# creating a new row with forward fill the gdpnom value
df2= df1['gdpnom'].ffill(axis=0)
df1['gdpnom']=df2
df_dayvisit = df1[df1['CountryName']=='seychelles']

df3 = pd.DataFrame(df_dayvisit, columns=['year', 'dayvisit'])
df3.plot(x ='year', y='dayvisit', kind = 'bar')
plt.show()
dt = df_dayvisit['dayvisit'] #took dayvisit column in dt dataframe

dt = dt.fillna(dt.mean()) # fill the null value using fillna. Mean()

df_dayvisit['dayvisit']=dt # putting the dt with all values again to
#df_dayvisit from where we nade dt


df1.update(df_dayvisit)
df_dayvisit = df[df['CountryName']=='bahrain']
df1 = pd.DataFrame(df_dayvisit, columns=['year', 'gdpnom'])
df1.plot(x ='year', y='gdpnom', kind = 'bar')
plt.show()
dt = df_dayvisit['dayvisit'] #took dayvisit column in dt dataframe

dt = dt.fillna(dt.ffill()) #using ffill function that is front fill
df_dayvisit['dayvisit']=dt
df.update(df_dayvisit)
df = pd.read_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx')


df_bahrain = df[df['CountryName']=='bahrain']

dt = df_bahrain['arrpleas'] #took dayvisit column in dt dataframe

dt = dt.fillna(dt.mean())
df_bahrain['arrpleas']=dt
df.update(df_bahrain)
print(dt)
df_bahrain = df[df['CountryName']=='bahrain']
#print(df_dayvisit)
df1 = pd.DataFrame(df_bahrain, columns=['year', 'arrbus'])
df1.plot(x ='year', y='arrbus', kind = 'bar')
plt.show()
dt = df_bahrain['arrbus'] #took arrbus column in dt dataframe

dt = dt.fillna(dt.mean())
df_bahrain['arrbus']=dt
df.update(df_bahrain)
print(dt)
df_bahrain = df[df['CountryName']=='bahrain']
df1 = pd.DataFrame(df_bahrain, columns=['year', 'arram'])
df1.plot(x ='year', y='arram', kind = 'bar')
plt.show()

dt = df_bahrain['arram'] #took dayvisit column in dt dataframe

dt = dt.fillna(dt.interpolate())
df_bahrain['arram']=dt
df.update(df_bahrain)

df_bahrain = df[df['CountryName']=='bahrain']
df1 = pd.DataFrame(df_bahrain, columns=['year', 'arreur'])
df1.plot(x ='year', y='arreur', kind = 'bar')
plt.show()
dt = df_bahrain['arreur']

dt = dt.fillna(dt.interpolate())
df_bahrain['arreur']=dt
df.update(df_bahrain)
df_bahrain = df[df['CountryName']=='bahrain']
df1 = pd.DataFrame(df_bahrain, columns=['year', 'arrger'])
df1.plot(x ='year', y='arrger', kind = 'bar')
plt.show()
dt = df_bahrain['arrger']

dt = dt.fillna(dt.interpolate())
df_bahrain['arrger']=dt
df.update(df_bahrain)
dt = df_bahrain['intxcac']

dt = dt.fillna(0)
df_bahrain['arrger']=dt
df.update(df_bahrain)

df_barbados = df[df['CountryName']=='bahrain']
dt = df_barbados['receipt']

dt = dt.fillna(dt.interpolate())
df_barbados['receipt']=dt
df.update(df_barbados)
df_bermuda = df[df['CountryName']=='bermuda']
df1 = pd.DataFrame(df_bermuda, columns=['year', 'arrpleas'])
df1.plot(x ='year', y='arrpleas', kind = 'bar')
plt.show()
dt = df_bermuda['arrpleas']

dt = dt.fillna(dt.interpolate())
df_bermuda['arrpleas']=dt
df.update(df_bermuda)
df_bermuda = df[df['CountryName']=='bermuda']
df1 = pd.DataFrame(df_bermuda, columns=['year', 'arrbus'])
df1.plot(x ='year', y='arrbus', kind = 'bar')
plt.show()
dt = dt.fillna(dt.interpolate())
df_bermuda['arrbus']=dt
df.update(df_bermuda)
df_maldives = df[df['CountryName']=='maldives']
df1 = pd.DataFrame(df_maldives, columns=['year', 'dayvisit'])
df1.plot(x ='year', y='dayvisit', kind = 'bar')
plt.show()
dt = df_maldives['dayvisit']

dt = dt.fillna(dt.interpolate())
df_maldives['dayvisit']=dt
df.update(df_maldives)
df_maldives = df[df['CountryName']=='maldives']
dt = df_maldives['arrbus']

dt = dt.fillna(0)
df_maldives['arrbus']=dt
df.update(df_maldives)
dt = df_maldives['arrwat']

dt = dt.fillna(0)
df_maldives['arrwat']=dt
df.update(df_maldives)
dt = df_maldives['arrger']

dt = dt.fillna(0)
df_maldives['arrger']=dt
df.update(df_maldives)
df.to_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx')
df_saotomeprincipe = df[df['CountryName']=='sao tome & principe']
df1 = pd.DataFrame(df_saotomeprincipe, columns=['year', 'ovnarriv'])
df1.plot(x ='year', y='ovnarriv', kind = 'bar')
plt.show()
dt = df_saotomeprincipe['ovnarriv']

dt = dt.fillna(dt.mean())
df_saotomeprincipe['ovnarriv']=dt
df.update(df_saotomeprincipe)
dt = df_saotomeprincipe['dayvisit']

dt = dt.fillna(0)
df_saotomeprincipe['dayvisit']=dt
df.update(df_saotomeprincipe)


dt = df_saotomeprincipe['arrger']

dt = dt.fillna(0)
df_saotomeprincipe['arrger']=dt
df.update(df_saotomeprincipe)
df_saotomeprincipe = df[df['CountryName']=='sao tome & principe']
df1 = pd.DataFrame(df_saotomeprincipe, columns=['year', 'arram'])
df1.plot(x ='year', y='arram', kind = 'bar')
plt.show()
dt = df_saotomeprincipe['arram']

dt = dt.fillna(dt.interpolate())
df_saotomeprincipe['arram']=dt
df.update(df_saotomeprincipe)
df_saotomeprincipe = df[df['CountryName']=='sao tome & principe']
dt = df_saotomeprincipe.fillna(df_saotomeprincipe.mean())
df.update(dt)
print(dt)
df_stKittsNevis = df[df['CountryName']=='st kitts and nevis']
dt = df_stKittsNevis['arrpleas']

dt = df_stKittsNevis.fillna(0)
df_stKittsNevis['arrpleas']=dt
df.update(df_stKittsNevis)
df_stKittsNevis = df[df['CountryName']=='st kitts and nevis']
dt = df_stKittsNevis['arrbus']

dt = df_stKittsNevis.fillna(0)
df_stKittsNevis['arrbus']=dt
df.update(df_stKittsNevis)
df_stKittsNevis = df[df['CountryName']=='st kitts and nevis']
dt = df_stKittsNevis['arrger']

dt = df_stKittsNevis.fillna(0)
df_stKittsNevis['arrger']=dt
df.update(df_stKittsNevis)

df_stKittsNevis = df[df['CountryName']=='st kitts and nevis']
df1 = pd.DataFrame(df_stKittsNevis, columns=['year', 'ovnarriv'])
df1.plot(x ='year', y='ovnarriv', kind = 'bar')
plt.show()
df_stKittsNevis = df[df['CountryName']=='st kitts and nevis']
df1 = pd.DataFrame(df_stKittsNevis, columns=['year', 'dayvisit'])
df1.plot(x ='year', y='dayvisit', kind = 'bar')
plt.show()
dt = df_stKittsNevis['dayvisit']

dt = dt.fillna(dt.interpolate())
df_stKittsNevis['dayvisit']=dt
df.update(df_stKittsNevis)

df_stKittsNevis = df[df['CountryName']=='st kitts and nevis']
df1 = pd.DataFrame(df_stKittsNevis, columns=['year', 'arrair'])
df1.plot(x ='year', y='arrair', kind = 'bar')
plt.show()
dt = df_stKittsNevis['arrair']

dt = dt.fillna(dt.mean())
df_stKittsNevis['arrair']=dt
df.update(df_stKittsNevis)

df_stKittsNevis = df[df['CountryName']=='st kitts and nevis']
df1 = pd.DataFrame(df_stKittsNevis, columns=['year', 'arrwat'])
df1.plot(x ='year', y='arrwat', kind = 'bar')
plt.show()
dt = df_stKittsNevis['arrwat']

dt = dt.fillna(dt.interpolate())
df_stKittsNevis['arrwat']=dt
df.update(df_stKittsNevis)
df_stKittsNevis = df[df['CountryName']=='st kitts and nevis']
df1 = pd.DataFrame(df_stKittsNevis, columns=['year', 'arram'])
df1.plot(x ='year', y='arram', kind = 'bar')
plt.show()
dt = df_stKittsNevis['arram']

dt = dt.fillna(dt.mean())
df_stKittsNevis['arram']=dt
df.update(df_stKittsNevis)
df_stKittsNevis = df[df['CountryName']=='st kitts and nevis']
df1 = pd.DataFrame(df_stKittsNevis, columns=['year', 'arreur'])
df1.plot(x ='year', y='arreur', kind = 'bar')
plt.show()
dt = df_stKittsNevis['arreur']

dt = dt.fillna(dt.mean())
df_stKittsNevis['arreur']=dt
df.update(df_stKittsNevis)
df_stKittsNevis = df[df['CountryName']=='st kitts and nevis']
df1 = pd.DataFrame(df_stKittsNevis, columns=['year', 'Arrivals'])
df1.plot(x ='year', y='Arrivals', kind = 'bar')
plt.show()
dt = df_stKittsNevis['Arrivals']

dt = dt.fillna(dt.mean())
df_stKittsNevis['Arrivals']=dt
df.update(df_stKittsNevis)
df_stLucia = df[df['CountryName']=='st lucia']
df1 = pd.DataFrame(df_stLucia, columns=['year', 'arrair'])
df1.plot(x ='year', y='arrair', kind = 'bar')
plt.show()
dt = df_stLucia['arrair']

dt = dt.fillna(dt.mean())
df_stLucia['arrair']=dt
df.update(df_stLucia)
df_stLucia = df[df['CountryName']=='st lucia']
df1 = pd.DataFrame(df_stLucia, columns=['year', 'arrwat'])
df1.plot(x ='year', y='arrwat', kind = 'bar')
plt.show()
dt = df_stLucia['arrwat']

dt = dt.fillna(dt.mean())
df_stLucia['arrwat']=dt
df.update(df_stLucia)

df_tonga = df[df['CountryName']=='tonga']
dt= df_tonga['receipt']

dt = dt.fillna(dt.mean())
df_tonga['receipt']=dt
df.update(df_tonga)
df_tonga = df[df['CountryName']=='tonga']
dt = df_tonga.fillna(df_tonga.mean())
df.update(dt)
df_singapore = df[df['CountryName']=='singapore']
df1 = pd.DataFrame(df_singapore, columns=['year', 'arrpleas'])
df1.plot(x ='year', y='arrpleas', kind = 'bar')
plt.show()
dt = df_singapore['arrpleas']

dt = dt.fillna(dt.mean())
df_singapore['arrpleas']=dt
df.update(dt)
df_singapore = df[df['CountryName']=='singapore']
df1 = pd.DataFrame(df_singapore, columns=['year', 'arrbus'])
df1.plot(x ='year', y='arrbus', kind = 'bar')
plt.show()
dt = df_singapore['arrbus']

dt = dt.fillna(dt.mean())
df_singapore['arrbus']=dt
df.update(dt)
df_singapore = df[df['CountryName']=='singapore']
df1 = pd.DataFrame(df_singapore, columns=['year', 'arrger'])
df1.plot(x ='year', y='arrger', kind = 'bar')
plt.show()
dt = df_singapore['arrger']

dt = dt.fillna(dt.interpolate())
df_singapore['arrger']=dt
df.update(dt)
df_trinidadandtobago = df[df['CountryName']=='trinidad and tobago']
dt = df_trinidadandtobago['Arrivals']

dt = dt.fillna(0)
df_trinidadandtobago['Arrivals']=dt
df.update(dt)
df = pd.read_excel('/Userss/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx')
df_trinidadandtobago = df[df['CountryName']=='trinidad and tobago']
dt = df_trinidadandtobago.fillna(df_trinidadandtobago.mean())
df.update(dt)
df = pd.read_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx')
df_solomonIslands = df[df['CountryName']=='solomon islands']
dt = df_solomonIslands['Arrivals']

dt = dt.fillna(0)
df_solomonIslands['Arrivals']=dt
df.update(df_solomonIslands)
df_solomonIslands = df[df['CountryName']=='solomon islands']
df1 = pd.DataFrame(df_solomonIslands, columns=['year', 'arrpleas'])
df1.plot(x ='year', y='arrpleas', kind = 'bar')
plt.show()
dt = df_solomonIslands['arrpleas']

dt = dt.fillna(dt.interpolate())
df_solomonIslands['arrpleas']=dt
df.update(df_solomonIslands)
df_solomonIslands = df[df['CountryName']=='solomon islands']
df1 = pd.DataFrame(df_solomonIslands, columns=['year', 'arrbus'])
df1.plot(x ='year', y='arrbus', kind = 'bar')
plt.show()
dt = df_solomonIslands['arrbus']

dt = dt.fillna(dt.interpolate())
df_solomonIslands['arrbus']=dt
df.update(df_solomonIslands)
df_solomonIslands = df[df['CountryName']=='solomon islands']
df1 = pd.DataFrame(df_solomonIslands, columns=['year', 'receipt'])
df1.plot(x ='year', y='receipt', kind = 'bar')
plt.show()
dt = df_solomonIslands['receipt']

dt = dt.fillna(dt.interpolate())
df_solomonIslands['receipt']=dt
df.update(df_solomonIslands)
df = pd.read_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx')
df_solomonIslands = df[df['CountryName']=='solomon islands']
dt = df_solomonIslands.fillna(df_solomonIslands.mean())
df.update(dt)
df.to_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx')

count = df.isna().sum()
print(count)



# implementing PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
df = pd.read_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx')

df1= df[['pop','areakm2','gdpnom','receipt','ovnarriv','dayvisit','arrpleas','arrbus','arrair','arrwat','arram','arreur','arrger','Arrivals','tcov','intxexg','intxexs','intxexal','intxcac','oteximg','otxims','otximal','otxcad']]

df_std = StandardScaler().fit_transform(df1)
mean_vec = np.mean(df_std,axis=0)
cov_mat = (df_std-mean_vec).T.dot((df_std-mean_vec))/(df_std.shape[0]-1)
cov_mat = np.cov(df_std.T)
eig_vals,eig_vecs = np.linalg.eig(cov_mat)
for i in range(len(eig_vals)):
    print(eig_vals[i]/sum(eig_vals))
per_var=[]
for i in range(len(eig_vals)):
    per_var.append((eig_vals[i]/sum(eig_vals))*100)

labels = ['PC' +str(x) for x in range(1,len(per_var)+1)]
plt.bar(x= per_var,height=per_var,tick_label =labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()


# pca implementation using PCA package

df = pd.read_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx')
df1 = df.drop(columns=['CountryName'],axis=1)


df_std = StandardScaler().fit_transform(df1)

pca = PCA(n_components=2)
X_new = pca.fit_transform(df_std)
pca_dd = pd.DataFrame(data = X_new , columns = ['principal component 1', 'principal component 2'])
print(pca_dd)
print(pca.explained_variance_ratio_)

plt.scatter(X_new[:,0], X_new[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()



def biplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()

biplot(X_new[:,0:2],np.transpose(pca.components_[0:2, :]))
plt.show()


# implementing Hierarchical clustering

import pandas as pd
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns; sns.set()

from mpl_toolkits.mplot3d import Axes3D

# reading
df = pd.read_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx')
df1 = df.drop(columns=['CountryName'],axis=1)


#normalising
data_scaled = normalize(df1)
data_scaled = pd.DataFrame(data_scaled, columns=df1.columns)
data_scaled.head()

# deciding the number of clusters
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.show()





plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.axhline(y=3, color='r', linestyle='--')
plt.show()




cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(data_scaled)
print(cluster.labels_)


plt.figure(figsize=(10, 7))
plt.scatter(data_scaled['year'], data_scaled['gdpnom'],data_scaled['dayvisit'], c=cluster.labels_,cmap='rainbow')

plt.show()


# visualizing plots

df = pd.read_excel('/Users/birparkash/PycharmProjects/untitled/world bank international arrivals islands fdraft.xlsx')

df_Singapore = df[df['CountryName'] == 'singapore']

x = df_Singapore['year']
a= df_Singapore['gdpnom']
b= np.log(df_Singapore['ovnarriv'])
df1 = pd.DataFrame({"x":x, "a":a, "b":b })

plt.plot( 'x', 'a', data=df1, marker='o', markerfacecolor='blue', color='skyblue', linewidth=4,label="dayvisit")
plt.plot( 'x', 'b', data=df1, marker='*', markerfacecolor='red',color='olive', linewidth=2,label="on-arrival")
plt.xlabel("Year")
plt.ylabel("gdpnom Vs OnArrival ")

plt.show()


df_SaoTome = df[df['CountryName'] == 'sao tome & principe']

x = df_SaoTome['year']
a= df_SaoTome['gdpnom']
b= np.log(df_SaoTome['ovnarriv'])
df1 = pd.DataFrame({"x":x, "a":a, "b":b })

plt.plot( 'x', 'a', data=df1, marker='o', markerfacecolor='blue', color='skyblue', linewidth=4,label="dayvisit")
plt.plot( 'x', 'b', data=df1, marker='*', markerfacecolor='red',color='olive', linewidth=2,label="on-arrival")
plt.xlabel("Year")
plt.ylabel("gdpnom Vs OnArrival ")

plt.show()
