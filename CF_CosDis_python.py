
## Read data from customer_invoice table
import pandas as pd
from scipy.spatial.distance import cosine

data = pd.read_csv('D:/Users/Keynes/Documents/Projects/Recommendation System/POS_Collaborative_v2.csv')



data.head(6).ix[:,0:5]

## Calculate distance matrix with cosine distance as pandas data frame

Data_Item = data.drop('Customer_number',1)

data_ibs = pd.DataFrame(index=Data_Item.columns,columns = Data_Item.columns)

data_ibs.head(4).ix[:,0:6]

for i in range(0,len(data_ibs.columns)):
    for j in range(0,len(data_ibs.columns)):
        data_ibs.ix[i,j] = 1 - cosine(Data_Item.ix[:,i],Data_Item.ix[:,j])
        

data_neigh = pd.DataFrame(index=data_ibs.columns,columns = range(1,21))

for i in range(0,len(data_ibs.columns)):
    data_neigh.ix[i,:20] = data_ibs.ix[0:,i].sort_values(ascending=False)[:20].index
    
data_neigh.head(6).ix[:10,2:20]

#User Based Recommendation Filter

#Formula for similarity Score

def getScore(history,similarity):
    return sum(history*similarity)/sum(similarity)
    
data_sims = pd.DataFrame(index=data.index,columns=data.columns)
data_sims.ix[:,:1] = data.ix[:,:1]

for i in range(0,len(data_sims.index)):
    for j in range(1,len(data_sims.columns)):
        user = data_sims.index[i]
        product = data_sims.columns[j]
        
        if data.ix[i][j] == 1:
            data_sims.ix[i][j] = 0
        else:
            product_top_names = data_neigh.ix[product][1:20]
            product_top_sims = data_ibs.ix[product].sort_values(ascending = False)[1:20]
            user_purchases = Data_Item.ix[user, product_top_names]
            
            data_sims.ix[i][j] = getScore(user_purchases,product_top_sims)
            
#Get top products
data_recommend = pd.DataFrame(index=data_sims.index,columns=['Customer_number','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
data_recommend.ix[0:,0] = data_sims.ix[:,0]

for i in range(0,len(data_sims.index)):
    data_recommend.ix[i,1:] = data_sims.ix[i,:].sort_values(ascending=False).ix[1:16,].index.transpose()
    
print(data_recommend.ix[:10,:5])
    

























