# Bank Customer Segmentation
> This project utilizes machine learning to preform customer bank segmentation. The ML techniques used here, are K-Means CLustering, Principle Component Analysis and Autoencoders. This project was done on the Google Collab Python notebook.

As part of a marketing department, ML and AI can be used to find the spending habits of various customers. This will allow the department to lauch a target marketing campaign towards those customers and maximise sales and profits. 

In this project, the marketing segmentation will be done for a data.The dataset used to train the ML/AI models in this project will be the bank's marketing data stored in the 'Marketing_data.csv' file. The columns of this dataset is shown in the following table.

|name of column | meaning  |
| ------ |---------------------- |
| CUSTID | Identification of Credit Card holder | 
| BALANCE | Balance amount left in customer's account to make purchases |
| BALANCE_FREQUENCY | How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated) |
| PURCHASES | Amount of purchases made from account |
|ONEOFFPURCHASES | Maximum purchase amount done in one-go |
|INSTALLMENTS_PURCHASES | Amount of purchase done in installment |
|CASH_ADVANCE | Cash in advance given by the user |
|PURCHASES_FREQUENCY | How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased) |
|ONEOFF_PURCHASES_FREQUENCY | How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased) |
|PURCHASES_INSTALLMENTS_FREQUENCY | How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done) |
|CASH_ADVANCE_FREQUENCY | How frequently the cash in advance being paid |
|CASH_ADVANCE_TRX | Number of Transactions made with "Cash in Advance" |
|PURCHASES_TRX | Number of purchase transactions made |
|CREDIT_LIMIT | Limit of Credit Card for user |
|PAYMENTS | Amount of Payment done by user |
|MINIMUM_PAYMENTS | Minimum amount of payments made by user|  
|PRC_FULL_PAYMENT | Percent of full payment paid by user |
|TENURE | Tenure of credit card service for user|

## K-Means Clustering

K-Means is an unsupervised ML algorithm that groups different data point together on a graph. The groups of datapoints are refered to as clusters and the variable 'K' is the number of those clusters. Each cluster is determined with a centroid.

The correct value of 'K' can be determined by the elbow method. To find the most optimal value of 'K', the WCSS(Within Cluster sum of Squares) can be calculated by the finding the sum of the sqaure of the producted the distance between each data point and it's distance to it's centroid. As you increase the clusters, the value of the WCSS goes down. The 'K'variable is determined a plot showing WCCS for each values of 'K'.  The optimal of 'K' is the one when the line for the WCCS starts flatting out.

The following code shows how the 'K' variable is determined using the KMeans method from the SkLearn class.

```python
scores_1 = []

range_values = range(1,20)

for i in range_values:
  kmeans = KMeans(n_clusters = i)
  kmeans.fit(creditcard_df_scaler)
  scores_1.append(kmeans.inertia_)
```  
## PCA (Principle Component Analysis)

PCA is also an unsupervised machine learnign algorithm, that performs dimension reductions. If the data used is scattered across more than 2 axis/components, then PCA aims to reduce the components to 2 while maining the same information.

The code below shows PCA being used to reduce the number of components for the data to 2.
```python
pca = PCA(n_components = 2)
principal_comp = pca.fit_transform(creditcard_df_scaler)
principal_comp
```
## Autoencoders
Autoencoders are a type of ANN(Artificial Neutral Networks) used to perform data encoding data. They use the same input data for the input and output. To do this they use a encoder which is the input data and a decoder network which is the output data and they are two ANNs opposing each other. The input and output images are taken to reduce its dimension to represent them in a much smaller space called the code layer but with the same information. Then they are upsampled using the decoder until the images are reconstructed again. The autoencoders will not work if all the input data are independant.

Here is the autoender network built for this project:
``` python
input_df = Input(shape = (17,))

x = Dense(7, activation='relu')(input_df)
x = Dense(500, activation='relu', kernel_initializer= 'glorot_uniform')(x)
x = Dense(500, activation='relu', kernel_initializer= 'glorot_uniform')(x)
x = Dense(2000, activation='relu', kernel_initializer= 'glorot_uniform')(x)

encoded = Dense(10, activation='relu', kernel_initializer='glorot_uniform')(x)

x = Dense(2000, activation='relu', kernel_initializer= 'glorot_uniform')(encoded)
x = Dense(500, activation='relu', kernel_initializer= 'glorot_uniform')(x)


decoded = Dense(17, kernel_initializer= 'glorot_uniform')(x)

#Autoencoder
autoencoder = Model(input_df, decoded)

#Encoder
encoder = Model(input_df, encoded)


autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')
```


