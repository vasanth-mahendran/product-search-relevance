import math
import pandas as pandas
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import time

start_time = time.time()
stemmer = PorterStemmer()

class product_search_relevance:

    # referenced kaggle scripts for using pandas in reading files
    def __init__(self):

        try:
            train = pandas.read_csv('data/train.csv', encoding="ISO-8859-1")
            test = pandas.read_csv('data/test.csv', encoding="ISO-8859-1")
            pro_desc = pandas.read_csv('data/product_descriptions.csv')
            attr = pandas.read_csv('data/attributes.csv')
            train_and_test = pandas.concat((train, test), axis=0, ignore_index=True)
        except Exception as error:
            print("problem while reading the files",repr(error))
            raise error

        print("--- Reading: %s minutes ---" % round(((time.time() - start_time) / 60), 2))

        if not(train is None or pro_desc is None or attr is None):
            try:
                brand = attr[attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(
                    columns={"value": "brand"})
                train_and_test = pandas.merge(train_and_test, pro_desc, how='left', on='product_uid')
                train_and_test = pandas.merge(train_and_test, brand, how='left', on='product_uid')
            except Exception as error:
                print("problem while merging the files", repr(error))
                raise error

        print("--- Merging: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
        self.N_Value = len(train_and_test.index)

        # lambda functions to apply stemming
        if not(train_and_test is None):
            try:
                train_and_test['search_term'] = train_and_test['search_term'].map(lambda x: self.pre_processing(x))
                train_and_test['product_title'] = train_and_test['product_title'].map(lambda x: self.pre_processing(x))
                train_and_test['product_description'] = train_and_test['product_description'].map(lambda x: self.pre_processing(x))
                train_and_test['brand'] = train_and_test['brand'].map(lambda x: self.pre_processing(x))
            except Exception as error:
                print("problem while applying stemming lambda functions on the files", repr(error))
                raise error

        print("--- Stemming: %s minutes ---" % round(((time.time() - start_time) / 60), 2))

        self.title_df = defaultdict(int)
        self.desc_df = defaultdict(int)
        training_records = train.shape[0]
        try:
            train_and_test['product_title'].apply(lambda x: self.buildtitledf(x))
            train_and_test['product_description'].apply(lambda x: self.builddescdf(x))
        except Exception as error:
            print("problem while applying df lambda functions on the files", repr(error))
            raise error

        print("--- Building Df: %s minutes ---" % round(((time.time() - start_time) / 60), 2))

        if not (train_and_test is None):
            try:
                train_and_test = train_and_test.merge(train_and_test.apply(self.tf_idf_score, axis=1), left_index=True, right_index=True)
                train = train_and_test[:training_records]
                co_efficient_1, co_efficient_2 = self.linear_regression(train['dp_value'], train['relevance'],len(train['dp_value']))
                train_and_test['predicted_score'] = train_and_test['dp_value'].map(lambda x: (x * co_efficient_1) + co_efficient_2)
            except Exception as error:
                print("problem while applying tf-idf lambda function", repr(error))
                raise error

        test = train_and_test[training_records:]
        print("--- TfIdf: %s minutes ---" % round(((time.time() - start_time) / 60), 2))
        #test.to_csv('df_test.csv', header=True, index=False, encoding='utf-8')
        pandas.DataFrame({"id": test['id'], "relevance": test['predicted_score']}).to_csv('result.csv', index=False)

    # Method to build df dictionary for title
    def buildtitledf(self,x):
        title_row = []
        title_set = x.split(" ")
        for word in title_set:
            if word not in title_row:
                title_row.append(word)
                self.title_df[word] += 1

    # Method to build df dictionary for description
    def builddescdf(self, x):
        desc_row = []
        desc_set = x.split(" ")
        for word in desc_set:
            if word not in desc_row:
                desc_row.append(word)
                self.desc_df[word] += 1

    # Refered this site for Linear Regression implementation
    # https://en.wikipedia.org/wiki/Simple_linear_regression
    # This is a simple work around
    # Re-implement this with Multiple Linear regression Equation
    # Apr-9 RMSE for Multiple Linear Regression is greater than Single Linear Regression
    # Hence Using Simple Linear Regression itself
    def linear_regression(self,cosine,relevance,l):
        cosine_squared_sum,product_sum,cosine_sum,relevance_sum = self.get_sums(cosine,relevance,l)
        prod_cos_relevance_sum = cosine_sum * relevance_sum
        covariance = self.get_covariance(product_sum,prod_cos_relevance_sum,l)
        variance = self.get_variance(cosine_squared_sum,cosine_sum,l)
        coefficient_1 = covariance/variance
        coefficient_2 = (relevance_sum - coefficient_1 * cosine_sum)/l
        return coefficient_1, coefficient_2

    def get_sums(self, cosine, relevance, length):
        product_sum = 0
        cosine_sum = 0
        relevance_sum = 0
        cosine_squared_sum = 0
        for i in range(length):
            cosine_sum += cosine[i]
            relevance_sum += relevance[i]
            product_sum += cosine[i] * relevance[i]
            cosine_squared_sum += cosine[i]*cosine[i]
        return cosine_squared_sum,product_sum,cosine_sum,relevance_sum

    # Method to calculate co-variance given products sum and prod relevance sum
    def get_covariance(self,sum_of_products,sum_product_cos_relevance,l):
        covariance = (sum_of_products - (sum_product_cos_relevance) / l)
        return covariance

    # Method to calculate variance given cosine square sum and cosine sum
    def get_variance(self,sum_cosine_squared,sum_cosine,l):
        variance = (sum_cosine_squared - ((sum_cosine * sum_cosine) / l))
        return variance

    # Using weight for desc, title, brand to combine cosine values into one
    # Discussed this with professor regarding combining the cosine values with weight
    # according to kaggle manual , product manufactured by the brand should be in
    # search result if brand name is searched
    # Hence brand has high weight when combining cosine values
    def tf_idf_score(self,row):
        search_set = row['search_term'].split(" ")
        se_vector = self.get_se_tf_vector(search_set)
        title_freq_vector = self.gettf_vector(row['product_title'])
        desc_freq_vector = self.gettf_vector(row['product_description'])
        title_vector = self.gettf_idf_title_vector(title_freq_vector)
        desc_vector = self.gettf_idf_desc_vector(desc_freq_vector)
        common = True if row['brand'] in search_set else False
        norm_title_vector = self.getnormvector(title_vector, search_set)
        norm_desc_vector = self.getnormvector(desc_vector, search_set)
        norm_se_term_vector = self.getnormvector(se_vector, search_set)
        dp_title_value = self.calculatedotproduct(norm_se_term_vector, norm_title_vector)
        dp_desc_value = self.calculatedotproduct(norm_se_term_vector, norm_desc_vector)
        dp_brand_value = (2 if common else 0)
        dp_value = dp_title_value+dp_desc_value+dp_brand_value
        return pandas.Series(dict(dp_value=dp_value))

    # Method to normalize the given vector using Euclidean distance
    def getnormvector(self, vector, se_set):
        euclidian_distance_sum = 0
        norm_vector = dict()
        for token, tf_idf_weight in vector.items():
            euclidian_distance_sum += math.pow(tf_idf_weight, 2)
        euclidian_distance = math.sqrt(euclidian_distance_sum)
        for se_term in se_set:
            normalized_tf_idf_weight = 0
            if se_term in vector.keys():
                if vector[se_term] > 0:
                    normalized_tf_idf_weight = (vector[se_term] / euclidian_distance)
                else:
                    normalized_tf_idf_weight = 0
            norm_vector[se_term] = normalized_tf_idf_weight
        return norm_vector

    # Method to build term frequency vector for search term
    def get_se_tf_vector(self,se_set):
        se_vector = dict()
        se_term_vector = defaultdict(int)
        for word in se_set:
            se_term_vector[word] += 1
        for word_tf, tf in se_term_vector.items():
            se_vector[word_tf] = 1 + math.log10(tf)
        return se_vector

    # Method to pre-process the string data i.e stemming,trimming
    def pre_processing(self,s):
        try:
            if isinstance(s, str):
                s = s.lower()
                escape_letters = ["$","  ","?",",","//","..","."," . "," / ","-"," \\"]
                for escape_letter in escape_letters:
                    s = s.replace(escape_letter," ")
                s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
                return s
            else:
                return " "
        except Exception as error:
            print("str causing error",s,repr(error))
            return " "

    # Method to calculate the dot product for given two vectors
    def calculatedotproduct(self, vector1, vector2):
        dotproduct = 0
        for key, value in vector1.items():
            if key in vector2.keys():
                dotproduct += value * vector2[key]
        return dotproduct

    # Method to build term frequency for given string
    def gettf_vector(self,str):
        tf_vector = defaultdict(int)
        word_set = str.split(" ")
        for word in word_set:
            tf_vector[word] += 1
        return tf_vector

    # Method to build tf-idf for given term frequency vector for title
    def gettf_idf_title_vector(self,tf_vector):
        tf_idf_title_vector = dict()
        for tf_word, t_freq in tf_vector.items():
            tf_weight = (1 + math.log10(t_freq))
            idf_weight = (math.log10(self.N_Value / self.title_df[tf_word]))
            tf_idf_weight = tf_weight * idf_weight
            tf_idf_title_vector[tf_word] = tf_idf_weight
        return tf_idf_title_vector

    # Method to build tf-idf for given term frequency vector for description
    def gettf_idf_desc_vector(self, tf_vector):
        tf_idf_desc_vector = dict()
        for tf_word, t_freq in tf_vector.items():
            tf_weight = (1 + math.log10(t_freq))
            idf_weight = (math.log10(self.N_Value / self.desc_df[tf_word]))
            tf_idf_weight = tf_weight * idf_weight
            tf_idf_desc_vector[tf_word] = tf_idf_weight
        return tf_idf_desc_vector

pd_search_obj = product_search_relevance()