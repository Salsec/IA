#Importation des librairys
import streamlit as st
import pandas as pd
import numpy as np
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from fpgrowth_py import fpgrowth


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
st.write(donnee[donnee['UnitePrice']==0.0])
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

st.title('Répartition du dataset')
#Division du dataset réorganisé en dataset d'entraînement , dataset de test et dataset de validation
with st.expander("Critère de répartition"):
    """Train_set : Celui-ci va être le plus volumineux en termes de donnée. En effet, 
    c’est sur ce jeu ci que le réseau va itérer durant la phase d’entrainement pour pouvoir s’approprier des paramètres, 
    et les ajuster au mieux. Certaines règles préconisent qu’il soit composé de 80% des données disponibles. 
    C’est la phase d’apprentissage.
    Validation_set : Quant à lui, on préconise d’avoir environ 10% des données disponible. 
    Ce jeu sera appelé une seule fois, à la fin de chaque itération d’entrainement. 
    Il va permettre d’équilibrer le système. C’est la phase d’ajustage.
    Test_set : Ce dernier va avoir un rôle bien différent des autres, 
    puisqu’il ne servira pas à ajuster notre réseau. En effet, 
    il va avoir pour rôle d’évaluer le réseau sous sa forme finale, 
    et de voir comment il arrive à prédire comme si le réseau était intégré à notre application. 
    C’est pour cela qu’il doit être composé exclusivement de nouveaux échantillons, 
    encore jamais utilisé pour éviter de biaiser les résultats en lui envoyant des donnés, 
    qu’il connaîtrait déjà et qu’il aurait déjà appris lors de la phase d’entrainement ou de validation. 
    Celui-ci encore peut être estimé de l’ordre de 10% des données disponible."""

#Division du dataset en Train_set soit 80% et X_test_validation soit 20%
#X_train represent les données d'entrainement, X_test_validation represente les
#données de test plus les données de validation
Train_set,X_test_validation = train_test_split(panier, test_size=0.2, random_state=42)
Test_set,Validation_set=train_test_split(X_test_validation, test_size=0.5, random_state=42)
#Visualisation du dataset d'entrainement
with st.expander("Visualisation du dataset d'entrainement"):
    st.write("Taille du dataset d'entrainement:", Train_set.shape)
    st.write(Train_set)

#Visualisation du dataset de validation
with st.expander("Visualisation du dataset de validation"):
    st.write("Taille du dataset de validation:", Validation_set.shape)
    st.write(Validation_set)

#Visualisation du dataset de test
with st.expander("Visualisation du dataset de test"):
    st.write("Taille du dataset de test:", Test_set.shape)
    st.write(Test_set)

st.title('Implémentation du model')

#Fonctionnement de l'algorithme FP-Croissance
with st.expander("Modélisation des règles d'association : algorithme de croissance Fp"):
    """Fp Growth est un modèle de Data Mining basé sur des règles d'association.
    Ce modèle permet, à partir d'un historique des transactions, 
    de déterminer l'ensemble des règles d'association les plus fréquentes dans le jeu de données. 
    Pour cela, il a besoin comme paramètre d'entrée de l'ensemble des transactions composé des paniers de produits que les clients ont déjà achetés.

    Étant donné un ensemble de données de transactions, 
    la première étape de la croissance FP consiste à calculer les 
    fréquences des articles et à identifier les articles fréquents.

    La deuxième étape de la croissance FP utilise une structure d'arborescence de suffixes (FP-tree) pour coder les transactions sans générer explicitement des ensembles candidats, 
    qui sont généralement coûteux à générer. Après la deuxième étape, 
    les itemsets fréquents peuvent être extraits de l'arbre FP et le modèle renvoie un ensemble de règles d'association de produits comme dans l'exemple ci-dessous :"""
    """{Produit A + Produit B} --> {Produit C} avec 60% de probabilité"""
    """{Produit B + Produit C} --> {Produit A + Produit D} avec une probabilité de 78 %"""
    """{Prodcut C} --> {Product B + Product D} avec 67% de probabilité"""
    """Pour établir ce tableau, il faut munir le modèle de 2 hyperparamètres :


    minSupRatio : support minimum pour qu'un itemset soit identifié comme fréquent. 
    Par exemple, si un élément apparaît 3 transactions sur 5, il a un support de 3/5=0,6.
    minConf :confiance minimale pour générer la règle d'association. 
    La confiance est une indication de la fréquence à laquelle une règle d'association s'est avérée vraie.
    Par exemple, si dans l'itemset des transactions X apparaît 4 fois, X et Y ne coexistent que 2 fois, 
    la confiance pour la règle X => Y est alors 2/4 = 0,5. 
    Le paramètre n'affectera pas l'exploration des ensembles d'éléments fréquents, 
    mais spécifiera la confiance minimale pour générer des règles d'association à partir d'ensembles d'éléments fréquents.
   
   
    Une fois les règles d'association calculées, 
    il ne vous reste plus qu'à les appliquer aux paniers produits des clients."""

#Génération des règles avec les données d'entainement

with st.expander("Règles générées"):
    freqItemSet, regles = fpgrowth(Train_set['StockCode'].values, minSupRatio=0.005, minConf=0.3)
    st.write('Nombre de règle générer:',len(regles))
 

#Conversion des règles générées en dataframe
with st.expander("Conversion des règles générées en dataframe"):
    lien=pd.DataFrame(regles,columns =['panier','prochain_produit','probabilite']) 
    lien=lien.sort_values(by='probabilite',ascending=False)
    st.write(lien)

#paramètre : element_panier = liste des éléments du panier du consommateur
#return : produit_suivant, probability = prochain produit à recommander, probabilité d'achat. 
# Ou (0,0) si aucun produit n'est trouvé.
#Description : depuis le panier d'un utilisateur, 
#renvoie le produit à recommander s'il n'a pas été trouvé
#dans la liste des associations du tableau associé au modèle FP Growth.
#Pour cela, nous recherchons dans le tableau des associations le produit à recommander de chaque

def calcul_produit_suivant_proba(element_panier):
    # pour chaque élément du panier du consommateur
    for i in element_panier:
        i={i}
         # si nous trouvons une association correspondante dans la table de croissance fp
        if len(lien[lien['panier']==i].values) !=0:
            # on prend le produit conséquent
             produit_suivant=list(lien[lien['panier']==i]['prochain_produit'].values[0])[0]
             # Nous vérifions que le client n'a pas déjà acheté le produit
             if produit_suivant not in element_panier:
                # Trouver la probabilité associée.
                probability=lien[lien['panier']==i]['probabilite'].values[0]
                return(produit_suivant,probability)
    #si aucun produit n'a été trouvé.
    return(0,0)

#Paramètre : panier = dataframe du panier du consommateur
#Retour : list_prochain_produit, list_probability = liste des prochains éléments à recommander et les probabilités d'achat associées.
#description : Fonction principale qui utilise celle ci-dessus. 
#Pour chaque client dans l'ensemble de données, nous recherchons un correspondant
#association dans le tableau du modèle Fp Growth. 
#Si aucune association n'est trouvée, nous appelons le compute_next_best_product
#fonction qui recherche des associations de produits individuels.
#Si aucune association individuelle n'est trouvée, la fonction renvoie (0,0).

def trouver_prochain_produit(panier):
    n=panier.shape[0]
    list_prochain_produit=[]
    list_probability=[]
    #pour chaque client
    for i in range(n):
        # panier client
        element=set(panier['StockCode'][i])
        # si on trouve une association dans le tableau de croissance fp correspondant à tout le panier du client.
        if len(lien[lien['panier']==element].values) !=0:
            # On prend le produit conséquent
            produit_suivant=list(lien[lien['panier']==element]['prochain_produit'].values[0])[0]
            # Probabilité associée dans le tableau
            probability=lien[lien['panier']==element]['probabilite'].values[0]
            list_prochain_produit.append(produit_suivant)
            list_probability.append(probability)
        # Si aucun antécédent à tout le panier n'a été trouvé dans le tableau
        elif len(lien[lien['panier']==element].values) ==0: 
            # fonction précédente
             produit_suivant,probability= calcul_produit_suivant_proba(panier['StockCode'][i])
             list_prochain_produit.append(produit_suivant)
             list_probability.append(probability)
    return(list_prochain_produit,list_probability)

#Calcul pour chaque client

# Ensemble de produits recommandés
list_prochain_produit, list_probability=trouver_prochain_produit(Train_set)
# Ensemble de probabilités associées
Train_set['Produits recommandes']=list_prochain_produit 
Train_set['Probability']=list_probability
#Visualisation des resultats
st.write(Train_set)

#### Calcul des prix estimés à partir des recommandations faites 
# et affichage du tableau final avec l'association (client, produit recommandé)

def prix_proba_liste(panier,data):
    panier=panier.rename(columns = {'StockCode': 'Panier client'})
    data_stock=data.drop_duplicates(subset ="StockCode", inplace = False)
    prix=[]
    description_list=[]
    for i in range(panier.shape[0]):
        stockcode=panier['Produits recommandes'][i]
        probability= panier['Probability'][i]
        if stockcode != 0:
            prix_unitaire=data_stock[data_stock['StockCode']==stockcode]['UnitPrice'].values[0]
            description=data_stock[data_stock['StockCode']==stockcode]['Description'].values[0]
            estim_price=prix_unitaire*probability
            prix.append(estim_price)
            description_list.append(description)
        else :
            prix.append(0)
            description_list.append('Null')
    return prix,description_list

Train_set['Prix estime'],Train_set['Product description']=prix_proba_liste(Train_set,donnee) 

Train_set = Train_set.reindex(columns=['Panier client','Produits recommandes','Description Produit','Probability','Price estimation'])
with st.expander("Vusialisation des résultats d'entrainement"):
    st.write(Train_set)
    #Evaluation du model
    st.write('En moyenne, le système de recommandation peut prédire en',Train_set['Probability'].mean()*100,  '%' 'du prochain produit que le client achètera' )

#validation#######################################################################
freqItemSet, regles = fpgrowth(Validation_set['StockCode'].values, minSupRatio=0.005, minConf=0.3)
lien=pd.DataFrame(regles,columns =['panier','prochain_produit','probabilite']) 
lien=lien.sort_values(by='probabilite',ascending=False)
#Calcul pour chaque client

# Ensemble de produits recommandés
list_prochain_produit1, list_probability1=trouver_prochain_produit(Validation_set)
# Ensemble de probabilités associées
Validation_set['Produits recommandes']=list_prochain_produit 
Validation_set['Probability']=list_probability
#Visualisation des resultats
#st.write(Validation_set)
Validation_set['Prix estime'],Validation_set['Product description']=prix_proba_liste(Validation_set,donnee) 

Validation_set = Validation_set.reindex(columns=['Panier client','Produits recommandes','Description Produit','Probability','Price estimation'])
with st.expander("Vusialisation des résultats de validation"):
    st.write(Validation_set)
    #Evaluation du model
    st.write('En moyenne, le système de recommandation peut prédire en',Validation_set['Probability'].mean()*100,  '%')

#Test #######################################################################
freqItemSet, regles = fpgrowth(Test_set['StockCode'].values, minSupRatio=0.005, minConf=0.3)
lien=pd.DataFrame(regles,columns =['panier','prochain_produit','probabilite']) 
lien=lien.sort_values(by='probabilite',ascending=False)
#Calcul pour chaque client

# Ensemble de produits recommandés
list_prochain_produit1, list_probability1=trouver_prochain_produit(Test_set)
# Ensemble de probabilités associées
Test_set['Produits recommandes']=list_prochain_produit 
Test_set['Probability']=list_probability
#Visualisation des resultats
#st.write(Validation_set)
Test_set['Prix estime'],Test_set['Product description']=prix_proba_liste(Test_set,donnee) 

Test_set = Test_set.reindex(columns=['Panier client','Produits recommandes','Description Produit','Probability','Price estimation'])
with st.expander("Vusialisation des résultats de validation"):
    st.write(Test_set)
    #Evaluation du model
    st.write('En moyenne, le système de recommandation peut prédire en',Test_set['Probability'].mean()*100,  '%')
