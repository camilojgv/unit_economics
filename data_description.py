import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

def basket(data):
    top_basket = list()
    # ----------
    for idx,i in enumerate(range(len(data))):
        string_list = data["TOP_BASKET"].iloc[i]
        string_type = type(string_list)
        try:
            if string_list != '[]':
                a_list = string_list.replace('\n','')
                b_list = a_list.replace(' ','')
            else:
                b_list = '["empty"]'
        except:
            b_list = '["empty"]'
        final_list = json.loads(b_list)
        top_basket.append(final_list[0])
    return top_basket

def delay(data):
    top_delay = list()
    # ----------
    for idx,i in enumerate(range(len(data))):
        string_list = data["TOP_ONTIME_TAG"].iloc[i]
        string_type = type(string_list)
        try:
            if string_list != '[]':
                a_list = string_list.replace('\n','')
                b_list = a_list.replace(' ','')
            else:
                b_list = '["empty"]'
        except:
            b_list = '["empty"]'
        final_list = json.loads(b_list)
        top_delay.append(final_list[0])
    return top_delay

def goal_cat(data):
    top_goal_cat = list()
    # ----------
    for idx,i in enumerate(range(len(data))):
        string_list = data["TOP_CATEGORIES"].iloc[i]
        string_type = type(string_list)
        try:
            if string_list != '[]':
                a_list = string_list.replace('\n','')
                b_list = a_list.replace(' ','')
            else:
                b_list = '["empty"]'
        except:
            b_list = '["empty"]'
        final_list = json.loads(b_list)
        top_goal_cat.append(final_list[0])
    return top_goal_cat

def normalization(data):
    x = list()
    for val in data: 
        is_nan = np.isnan(val)
        if not is_nan:
            x.append(val)
    x_min = min(x)
    x_max = max(x)
    x_normal = (data-x_min)/(x_max-x_min)
    return x_normal

def standarization(data):
    x = list()
    for val in data: 
        is_nan = np.isnan(val)
        if not is_nan:
            x.append(val)
    mu = np.mean(x)
    sigma = np.var(x)
    x_std = (data - mu)/ sigma
    return x_std

def plot_data(X_norm):
        # f, ax = plt.subplots(figsize=(10, 8))
        # sbn.heatmap(X_norm.corr(), mask=np.zeros_like(X_norm.corr(), dtype=np.bool), cmap=sbn.diverging_palette(220, 10, as_cmap=True),
        #         square=True, ax=ax)
        # plt.show()
        # f, ax = plt.subplots(figsize=(10, 8))

        # sbn.pairplot(X_norm, x_vars=['TOTAL_WEIGHT','AVG_UNITS'],y_vars=['UNIT_ECONOMICS'],hue='TOP_CATEGORIES_CLEAN')
        # plt.show()

        sbn.heatmap(X_norm.corr(),annot=True,lw=1)
        plt.show()

if __name__ == '__main__':
    data = pd.read_csv('ue_models.csv')
    data['TOP_BASKET_CLEAN'] = basket(data)
    data['TOP_CATEGORIES_CLEAN'] = goal_cat(data)
    data['TOP_DELAY_CLEAN'] = delay(data)
    cols = data.columns
    X = data.filter(items=['AVG_DATE_CREATED','AVG_UNITS','AVG_REFERENCES',
                            'DISCOUNT_RATE','CANCELING_RATE','HUNTER_SALE_RATE','AVG_ORDER_VALUE',
                            'AVG_CANCEL_VALUE','TOP_BASKET_CLEAN','TOP_DELAY_CLEAN','DELAY_MEAN', 
                            'NON_PARETO_RATE','TOTAL_WEIGHT','AVG_LEAD_TIME',
                            'TOP_CATEGORIES_CLEAN', 'UNIT_ECONOMICS'])
    Y_log = data.filter(items='IS_PROFITABLE')
    Y_real = data.filter(items='UNIT_ECONOMICS')

    X_norm = X.copy()
    X_std = X.copy()
    for col in X_norm.columns:
        if X_norm[col].dtypes == float:
            X_norm[col] = normalization(X_norm[col].values)
    
    for col in X_std.columns:
        if X_std[col].dtypes == float:
            X_std[col] = standarization(X_std[col])
    X_norm = pd.get_dummies(data=X_norm, drop_first=True)
    final_X = X_norm.drop(['TOP_DELAY_CLEAN_empty', 'TOP_BASKET_CLEAN_empty', 'TOP_CATEGORIES_CLEAN_empty'], axis=1)
    final_X.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = final_X.replace(np.nan, 0)
    X_train, X_test, y_train, y_test = train_test_split(data, Y_log, test_size=0.25, random_state=101)
    print(y_train.T)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    plot_data(X)    #model = LogisticRegression()
    #model.fit(X_train.values, y_train)
    #print('Intercept -> {}'.format(model.intercept_))
    #coeff_parameter = pd.DataFrame(model.coef_, X_train.columns,columns=['Coefficient'])
    #predictions = model.predict(X_test)
    #score = model.score(X_test, y_test)
    #plt.figure(figsize=(9,9))
    # cm = metrics.confusion_matrix(y_test, predictions)
    # sbn.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    # plt.ylabel('Actual label')
    # plt.xlabel('Predicted label')
    # all_sample_title = 'Accuracy Score: {0}'.format(score)
    # plt.title(all_sample_title, size = 15)
    # plt.show()
