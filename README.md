# product-search-relevance

This is the kaggle Participation script. Refer the link to know the kaggle competition https://www.kaggle.com/c/home-depot-product-search-relevance

## Introduction
Developing a model that can accurately predict the relevance of search results to improve the customer shopping experience. Input for the model is the human rated relevance for the search term and product. Model has to mimic the human raters hence predicting the relevance for the remaining test data.

## Data Preprocessing
Data has been given across multiple files like train.csv, product_descriptions.csv, attributes.csv and test.csv for which relevance need to be predicted. Since all files share a unique primary field ‘product-id’, all the data files can be merged into single file.

Data requires below preprocessing
•	Stemming
•	Trimming (double space, few other escape literals)
Since preprocessing will reduce the amount data to be processed further for accuracy and memory constraint.
TF-IDF Vector Space Model
Each search term is considered as Query Vector and description, title & brand are considered as Document Vectors.
 
TFterm, d -IDF = TFterm, d*idf

idfterm = Log10(N/df)

N- total number of documents 
df- total number of document where term occurs
TFterm, d = 1+Log10(tfterm, d)

N- total number of documents

df- total number of document where term occurs 
tfterm-frequency of the term in the document d

Document Vectors has been built using TF-IDF weights and Query Vectors has been built using TF weight. Both Query and Documents vectors are normalized and dot product of these two gives cosine similarities.

## Basic Linear Regression Model
Co-efficient ‘a’ and ‘b’ can be found from two columns values ‘cosine value’ and ‘relevance’ of train data using below formula.

## Predicting the relevance
Relevance for the test data can be predicted from the linear equation using the linear regression co-efficient and cosine values. Finally, all the predictions for the test data has to be exported to csv file for the submission.

Relevance prediction = a * cosine value + b

a, b – linear regression coefficients

## Libraries used
•	Pandas (for processing data files)
•	Nltk (for stemming)
