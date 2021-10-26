import pandas as pd
import numpy as np


def prefilter_items(data, item_features=None, take_n_popular=5000):
    # Уберем самые популярные товары (их и так купят)
    popularity = (data.groupby('item_id')['user_id'].nunique() / data['user_id'].nunique()).reset_index()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    week_year_ago = data['week_no'].max() - 52
    data = data[data['week_no'] - week_year_ago > 0]
    # Уберем не интересные для рекоммендаций категории (department)
    # product_info = pd.read_csv('../data/product.csv')
    # item_features.rename(columns={'PRODUCT_ID': 'item_id',
    #                              'DEPARTMENT': 'product_category'},
    #                     inplace=True)
    data = data.merge(item_features[['item_id', 'department']], on='item_id')
    prod_cat_freqs = data.groupby('department')['user_id'].count()
    prod_cat_freqs = prod_cat_freqs[prod_cat_freqs >= 100].index.to_list()
    data = data[data['department'].isin(prod_cat_freqs)]
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    # Пусть 1 в столбце sales_value равна 60 руб, тогда оставим только товары дороже 60 руб
    data = data[data['sales_value'] > 1]
    # Уберем слишком дорогие товары
    too_expensive = data.sales_value.quantile(0.9999)
    data = data[data['sales_value'] < too_expensive]
    top = popularity.sort_values('share_unique_users', ascending=False).head(take_n_popular)['item_id'].to_list()
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999
    return data


def postfilter_items(user_id, recommednations):
    pass


if __name__ == '__main__':
    data = pd.read_csv('../data/transaction_data.csv')
    product_info = pd.read_csv('../data/product.csv')
    data.columns = [col.lower() for col in data.columns]
    data.rename(columns={'household_key': 'user_id',
                         'product_id': 'item_id'},
                inplace=True)

    test_size_weeks = 3

    data_train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
    data_test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]

    data_train = prefilter_items(data_train, product_info)
