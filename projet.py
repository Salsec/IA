#Importation des librairys
import streamlit as st
import pandas as pd
import numpy as np
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report



st.title('SYSTEME DE RECOMMANDATION')

#chargement du dataset
DATA_URL = ('data.csv')

def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL,encoding = 'unicode_escape')
    #Ajout de la colonne "GroupPrice"
   
    return data

data_load_state = st.text('Chargement des données.......')
data = load_data(DATA_URL)
data_load_state.text("Fais!.........................")
#Affichage de la taille du dataset
st.write('Nombre de ligne du dataset:', data.shape[0])
st.write('Nombre de colonne du dataset:', data.shape[1])

#Affichage des données brutes du dataset
with st.expander("Afficher les données brutes"):
    st.subheader('Données brutes')
    st.write(data)

# generation du rapport de donnée du dataset
with st.expander("Rapport de donnée du dataset"):
    profil=pandas_profiling.ProfileReport(data, title="Rapport de dataset avant traitement")
    st_profile_report(profil)


st.title('Exploration des données')

donnee=data

#Si il existe, visualisation en detail et traitement si non on ne fait rien
def verify_values(data):
    if data.isnull().values.any():
        res="Oui"
    else:
        res="Non"
    return res

#Verification de la pésence de données négatives ou égale à zéro dans la colonne Quantity
def verify_quantity(donnee):
    if len(donnee[donnee['Quantity'] <= 0].index)>0:
        res="Oui"
    else:
        res="Non"
    return res

with st.expander("Verification de la pésence de valeurs négative dans la colonne Quantity"):
    st.write('Existe il des données négatives ou égale à zéro dans la colonne Quantity? ', verify_quantity(donnee))

    #calcule et affichage du nombre de ligne contenants des données négatives ou égale à zéro dans la colonne Quantity
    st.write('Le nombre de ligne contenant des quantités inférieur ou égale zéro :',len(donnee[donnee['Quantity'] <= 0].index))

    #Visualisation des lignes contenants de données négatives ou égale à zéro dans la colonne Quantity
    if st.checkbox('Afficher les données négatives ou égale à zéro dans la colonne Quantity'):
        st.subheader('Visualisation des donnéesnégatives ou égale à zéro dans la colonne Quantity du dataset')
        st.write(donnee[donnee['Quantity'] <= 0])

#traitement
#Conversionen valeur positive des quantités négative 
donnee['Quantity']=donnee['Quantity'].abs()

with st.expander("Verification de la pésence de valeurs négative dans la colonne Quantity après traitement"):
    st.write('Existe il des données négatives ou égale à zéro dans la colonne Quantity après traitement? ', verify_quantity(donnee))

    #calcule et affichage du nombre de ligne contenants des données négatives ou égale à zéro dans la colonne Quantity
    st.write('Le nombre de ligne contenant des quantités inférieur ou égale zéro :',len(donnee[donnee['Quantity'] <= 0].index))



#Verification de la pésence de ligne contenant de données négatives ou égale à zéro dans la colonne Uniteprice
def verify_unitprice(donnee):
    if len(donnee[donnee['UnitPrice'] == 0.0].index)>0:
        res="Oui"
    else:
        res="Non"
    return res
with st.expander('Verification de la pésence de valeur abérants dans la colonne Uniteprice'):
    st.write('Existe il des données négatives ou égale à zéro dans la colonne UnitPrice? ', verify_unitprice(donnee))

    #calcule et affichage du nombre de ligne contenants des données négatives ou égale à zéro dans la colonne UnitPrice
    st.write('Le nombre de ligne contenant des quantités inférieur ou égale zéro :',len(donnee[donnee['UnitPrice'] == 0].index))

    #Visualisation des lignes contenants de données négatives ou égale à zéro dans la colonne UnitPrice
    if st.checkbox('Afficher les lignes contenants des valeur négatives ou égale à zéro dans la colonne UnitPrice'):
        st.subheader('Visualisation des données négatives ou égale à zéro dans la colonne UnitPrice du dataset')
        st.write(donnee[donnee['UnitPrice'] == 0.0])

#Traitement des ligne contenant des UnitPrice à 0
#Regroupement de chaque produit avec ces différent prix 
def somme_quantity_par_produit(StockCode,UnitPrice):
    tab=donnee.groupby([StockCode]).agg({ UnitPrice: lambda i: list(i)})
    return tab

tab4='StockCode'
tab3='UnitPrice'
tab=somme_quantity_par_produit(tab4,tab3)

#calcul du prix moyen pour chaque produit
for i in range(len(tab)):
    tab.loc[tab.index[i],'UnitPrice']=np.mean(tab.loc[tab.index[i],'UnitPrice'])
    format(tab.loc[tab.index[i],'UnitPrice'], '.2f')
indice = donnee[donnee['UnitPrice'] == 0.0]
tab=tab.reset_index()


#remplissage du UnitePrice à zéro par la valeur moyenne du prix du produit correspondante

for h in indice.index:
    code_pro=donnee.loc[donnee.index[int(h)],'StockCode']
    vrai_ind=tab[tab['StockCode']==code_pro].index
    donnee.loc[donnee.index[int(h)],'UnitPrice']=tab.loc[tab.index[vrai_ind[0]],'UnitPrice']

#Traiment
#Suppression des ligne contenant des UnitPrice égales à zéro apès traitement
donnee=donnee.drop(donnee[donnee['UnitPrice'] == 0.0].index)

#Verification de la présence des ligne contenant des UnitPrice égales à zéro apès traitement
with st.expander('Verification de la pésence de valeur abérants dans la colonne Uniteprice apès traitement'):
    st.write('Existe il des données négatives ou égale à zéro dans la colonne UnitPrice apès traitement? ', verify_unitprice(donnee))

    #calcule et affichage du nombre de ligne contenants des données négatives ou égale à zéro dans la colonne UnitPrice
    st.write('Le nombre de ligne contenant des quantités inférieur ou égale zéro :',len(donnee[donnee['UnitPrice'] == 0].index))

   


#Verrification des données manquantes
with st.expander("Verification des données manquantes"):
    st.write('Exist il des données manquantes dans le dataset',verify_values(data))

    #Visulalisation des colonnes contenants des valeurs manquantes
    st.write('Visulalisation des colonnes contenants des valeurs manquantes')

    #nombre de valeurs manquantes pour chaque colonnes
    somme_par_colonne=data.isnull().sum()
    st.write(somme_par_colonne[somme_par_colonne.values>0])

    #Visualisation des données manquantes du dataset
    donnee_manquante=data[data.isnull().any(axis=1)]
    if st.checkbox('Afficher les données manquantes'):
        st.subheader('Visualisation des données manquantes du dataset')
        st.write(donnee_manquante)


#Traitement
#Suppression des ligne contenant des CustomerID manquante car on ne peut pas traité ces donnée
donnee=data.dropna()

#Verrification des données manquantes après traitement
with st.expander("Verification des données manquantes après traitement"):
    st.write('Exist il des données manquantes dans le dataset après traitement?',verify_values(donnee))
    #nombre de valeurs manquantes pour chaque colonnes après traitement
    somme_par_colonne=donnee.isnull().sum()
    st.write(somme_par_colonne[somme_par_colonne.values>0])

donnee['GroupPrice']=donnee['Quantity']*donnee['UnitPrice']
#Affichage des données du dataset après suppression des valaeurs abérants et manquantes
with st.expander('Visualisation des données nouveau du dataset'):
    st.write(donnee.shape)
    st.write(donnee)

# generation du rapport de donnée du dataset après traitement
with st.expander("Rapport de donnée du dataset après traitement"):
    profil=pandas_profiling.ProfileReport(donnee, title="Rapport de dataset après traitement")
    st_profile_report(profil)

#Regroupement tous les produits qu'un client a achetés ensemble.
#Chaque ligne correspond à une transaction composée du numéro de facture, de l'identifiant client et de tous les produits achetés.
panier = donnee.groupby(['InvoiceNo','CustomerID']).agg({'StockCode': lambda s: list(set(s))})
with st.expander('Reorganisation et normalisation du dataset'):
    st.write(panier.shape)
    st.write(panier)

# generation du rapport de donnée du dataset après reorganisation
with st.expander("Rapport de donnée du dataset après reorganisation"):
    profil=pandas_profiling.ProfileReport(panier, title="Rapport de dataset après reorganisation")
    st_profile_report(profil)