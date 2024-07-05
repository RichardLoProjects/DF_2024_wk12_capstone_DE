import streamlit as st
import pandas as pd
import psycopg2 as psql
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

class MktIndexPortfolioItemStruct:
    def __init__(self, name, uid, weight):
        self.name = name
        self.uid = uid
        self.weight = weight
    def set_df(self, df):
        self.df = df

@st.cache_data
def pinging_database():
    conn = psql.connect(
        database = st.secrets['DATABASE_NAME']
        , host = st.secrets['DATABASE_HOST']
        , port = int(st.secrets['DATABASE_PORT'])
        , user = st.secrets['SQL_USERNAME']
        , password = st.secrets['SQL_PASSWORD']
    )
    s_name = st.secrets['SQL_TABLENAME_FOR_STATIC_DATA']
    d_name = st.secrets['SQL_TABLENAME_FOR_DYNAMIC_DATA']
    cur = conn.cursor()
    cur.execute(f'SELECT * FROM {s_name};')
    s_data = cur.fetchall()
    s_cols = [desc[0] for desc in cur.description]
    s_df = pd.DataFrame(s_data, columns=s_cols)
    cur.execute(f'SELECT * FROM {d_name};')
    d_data = cur.fetchall()
    d_cols = [desc[0] for desc in cur.description]
    d_df = pd.DataFrame(d_data, columns=d_cols)
    conn.close()
    return s_df, d_df

def draw_homepage_tab(df, name, uid):
    cols = ['item_id', 'price_timestamp', 'avg_high_price', 'avg_low_price', 'avg_mid_price', 'avg_micro_price', 'total_volume']
    df2 = df[(df['item_id']==uid)][cols].sort_values('price_timestamp', ascending=True).tail(2)
    st.header(name)
    fig = plt.figure(figsize=(10, 6))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label = 'Midprice'
            , value = f"{int(df2['avg_mid_price'].iloc[1]):,}"
            , delta = f"{int(df2['avg_mid_price'].iloc[1]) - int(df2['avg_mid_price'].iloc[0]):,}"
        )
    with c2:
        st.metric(
            label = 'Microprice'
            , value = f"{int(df2['avg_micro_price'].iloc[1]):,}"
            , delta = f"{int(df2['avg_micro_price'].iloc[1]) - int(df2['avg_micro_price'].iloc[0]):,}"
        )
    with c3:
        st.metric(
            label = 'Volume'
            , value = f"{int(df2['total_volume'].iloc[1]):,}"
            , delta = f"{int(df2['total_volume'].iloc[1]) - int(df2['total_volume'].iloc[0]):,}"
        )
    df3 = df[(df['item_id']==uid)][cols].sort_values('price_timestamp', ascending=True)
    show_mid = st.checkbox('Plot Mid', key=f'{name}: mid')
    show_micro = st.checkbox('Plot Micro', key=f'{name}: micro')
    if show_mid:
        plt.plot(df3['price_timestamp'], df3[['avg_mid_price']])
        plt.ylabel('Price', fontsize=15)
    if show_micro:
        plt.plot(df3['price_timestamp'], df3[['avg_micro_price']])
        plt.ylabel('Price', fontsize=15)
    if not any([show_mid, show_micro]):
        plt.bar(df3['price_timestamp'].tolist(), df3['total_volume'].tolist(), width=0.02)
        plt.ylabel('Volume', fontsize=15)
    plt.xlabel('Date', fontsize=15)
    plt.gca().ticklabel_format(axis='y', style='plain')
    st.pyplot(fig)

def home_page(df):
    st.write(' ')
    t1, t2, t3, t4, t5 = st.tabs(['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5'])
    with t1:
        draw_homepage_tab(df, 'Old school bond', 13190)
    with t2:
        draw_homepage_tab(df, 'Twisted bow', 20997)
    with t3:
        draw_homepage_tab(df, 'Scythe of vitur (uncharged)', 22486)
    with t4:
        draw_homepage_tab(df, 'Tumeken\'s shadow (uncharged)', 27277)
    with t5:
        draw_homepage_tab(df, 'Zaryte crossbow', 26374)

def draw_marketindex_page_tab(name, df, portfolio, s):
    cols = ['price_timestamp', 'weighted_sum_prices']
    df2 = df[cols].sort_values('price_timestamp', ascending=True)
    df3 = df2.tail(2)
    st.header(name)
    items = [{'name': item.df.iloc[0]['item_name'], 'weight': item.weight} for item in portfolio]
    df4 = pd.concat([pd.DataFrame.from_dict([item], orient='columns') for item in items], ignore_index=True)
    fig = plt.figure(figsize=(10, 6))
    st.metric(
        label = 'Index Price'
        , value = f"{int(df3['weighted_sum_prices'].iloc[1]):,}"
        , delta = f"{int(df3['weighted_sum_prices'].iloc[1]) - int(df3['weighted_sum_prices'].iloc[0]):,}"
    )
    plt.plot(df2['price_timestamp'], df2[['weighted_sum_prices']])
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Price (Weighted Avg)', fontsize=15)
    plt.gca().ticklabel_format(axis='y', style='plain')
    st.pyplot(fig)
    st.write('Portfolio made from the following weighted sum:')
    st.dataframe(df4.set_index('name'))

def marketindex_page(df,s):
    st.write(' ')
    r1_portfolio = [
        MktIndexPortfolioItemStruct('Dex', 21034, 20)
        , MktIndexPortfolioItemStruct('Arcane', 21079, 20)
        , MktIndexPortfolioItemStruct('Buckler', 21000, 4)
        , MktIndexPortfolioItemStruct('Dhcb', 21012, 4)
        , MktIndexPortfolioItemStruct('Din', 21015, 3)
        , MktIndexPortfolioItemStruct('Anc Hat', 21018, 3)
        , MktIndexPortfolioItemStruct('Anc Top', 21021, 3)
        , MktIndexPortfolioItemStruct('Anc Leg', 21024, 3)
        , MktIndexPortfolioItemStruct('Claws', 13652, 3)
        , MktIndexPortfolioItemStruct('Kodai Insig', 21043, 2)
        , MktIndexPortfolioItemStruct('Elder Maul', 21003, 2)
        , MktIndexPortfolioItemStruct('Tbow', 20997, 2)
    ]
    r2_portfolio = [
        MktIndexPortfolioItemStruct('Avernic', 22477, 8)
        , MktIndexPortfolioItemStruct('Ghrazi', 22324, 2)
        , MktIndexPortfolioItemStruct('Sang', 22481, 2)
        , MktIndexPortfolioItemStruct('Justi Helm', 22326, 2)
        , MktIndexPortfolioItemStruct('Justi Body', 22327, 2)
        , MktIndexPortfolioItemStruct('Justi Legs', 22328, 2)
        , MktIndexPortfolioItemStruct('Scy', 22486, 1)
    ]
    r3_portfolio = [
        MktIndexPortfolioItemStruct('Lb', 25975, 7)
        , MktIndexPortfolioItemStruct('Fang', 26219, 7)
        , MktIndexPortfolioItemStruct('Ward', 25985, 3)
        , MktIndexPortfolioItemStruct('Masori Mask', 27226, 2)
        , MktIndexPortfolioItemStruct('Masori Body', 27229, 2)
        , MktIndexPortfolioItemStruct('Masori Legs', 27232, 2)
        , MktIndexPortfolioItemStruct('Staff', 27277, 1)
    ]
    portfolio_list = [r1_portfolio, r2_portfolio, r3_portfolio]
    portfolio_df_list = []
    for lst in portfolio_list:
        new_df = df[['price_timestamp', 'item_name']].copy().drop_duplicates()
        for item in lst:
            item_df = df[(df['item_id']==item.uid)][df.columns.to_list()]
            item_df.sort_values('price_timestamp', ascending=True, inplace=True)
            price = item_df['avg_low_price']
            item_df[f'{item.name}_taxed_price'] = np.where(
                price.notna()
                , price-np.minimum(5_000_000,price//100)
                , 0
            )
            item.set_df(item_df[['price_timestamp', 'item_name', f'{item.name}_taxed_price']])
        new_df['weighted_sum_prices'] = 0
        for item in lst:
            new_df = pd.merge(new_df,item.df[['price_timestamp',f'{item.name}_taxed_price']],'inner',on='price_timestamp')
            new_df['weighted_sum_prices'] += new_df[f'{item.name}_taxed_price'] * item.weight
        new_df['weighted_sum_prices'] //= sum(item.weight for item in lst)
        new_df.sort_values('price_timestamp', inplace=True)
        portfolio_df_list.append(new_df)
    t1, t2, t3 = st.tabs(['Raids 1 Index', 'Raids 2 Index', 'Raids 3 Index'])
    with t1:
        draw_marketindex_page_tab('Chambers of Xeric Index', portfolio_df_list[0], r1_portfolio, s)
    with t2:
        draw_marketindex_page_tab('Theatre of Blood Index', portfolio_df_list[1], r2_portfolio, s)
    with t3:
        draw_marketindex_page_tab('Tombs of Amascut Index', portfolio_df_list[2], r2_portfolio, s)

def draw_queryitem_page(df, name, uid):
    cols = ['item_id', 'price_timestamp', 'avg_high_price', 'avg_low_price', 'avg_mid_price', 'avg_micro_price', 'total_volume']
    df2 = df[(df['item_id']==uid)][cols].sort_values('price_timestamp', ascending=True).tail(2)
    st.header(name)
    fig = plt.figure(figsize=(10, 6))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label = 'Midprice'
            , value = f"{int(df2['avg_mid_price'].iloc[1]):,}"
            , delta = f"{int(df2['avg_mid_price'].iloc[1]) - int(df2['avg_mid_price'].iloc[0]):,}"
        )
    with c2:
        st.metric(
            label = 'Microprice'
            , value = f"{int(df2['avg_micro_price'].iloc[1]):,}"
            , delta = f"{int(df2['avg_micro_price'].iloc[1]) - int(df2['avg_micro_price'].iloc[0]):,}"
        )
    with c3:
        st.metric(
            label = 'Volume'
            , value = f"{int(df2['total_volume'].iloc[1]):,}"
            , delta = f"{int(df2['total_volume'].iloc[1]) - int(df2['total_volume'].iloc[0]):,}"
        )
    df3 = df[(df['item_id']==uid)][cols].sort_values('price_timestamp', ascending=True)
    graph_choice = st.radio('Graph type: ', options=['Mid Price', 'Micro Price', 'Volume', 'KDE'])
    match graph_choice:
        case 'Mid Price':
            plt.plot(df3['price_timestamp'], df3[['avg_mid_price']])
            plt.ylabel('Price', fontsize=15)
            plt.xlabel('Date', fontsize=15)
            plt.gca().ticklabel_format(axis='y', style='plain')
        case 'Micro Price':
            plt.plot(df3['price_timestamp'], df3[['avg_micro_price']])
            plt.ylabel('Price', fontsize=15)
            plt.xlabel('Date', fontsize=15)
            plt.gca().ticklabel_format(axis='y', style='plain')
        case 'Volume':
            plt.bar(df3['price_timestamp'].tolist(), df3['total_volume'].tolist(), width=0.02)
            plt.ylabel('Volume', fontsize=15)
            plt.xlabel('Date', fontsize=15)
            plt.gca().ticklabel_format(axis='y', style='plain')
        case 'KDE':
            df4 = df3.dropna(subset=['avg_mid_price'])
            df4['price_delta'] = df4['avg_mid_price'].diff()
            price_deltas = df4['price_delta'].dropna().values
            data_kde = kde.gaussian_kde(price_deltas)
            x = np.linspace(min(price_deltas), max(price_deltas), 400)
            density = data_kde(x)
            plt.plot(x, density)
            plt.xlabel('Price Delta', fontsize=15)
            plt.ylabel('Density', fontsize=15)
            plt.gca().ticklabel_format(axis='x', style='plain')
    st.pyplot(fig)

def queryitem_page(df, s):
    st.write(' ')
    #use_id = st.checkbox('Query with item ID instead of item name')
    #user_input = st.number_input('Enter item ID:', step=1)# if use_id else st.text_input('Enter item name:')
    user_input = st.number_input('Enter an item ID:', step=1)
    if user_input:
        filtered_df = s[(s['item_id']==user_input)][['item_id', 'item_name']]
        if filtered_df.empty:
            st.write('No item found matching your search.')
        else:
            name = filtered_df.iloc[0]['item_name']
            draw_queryitem_page(df, name, user_input)

def draw_correlation(df, id1, id2):
    font_size = 10
    cols = ['item_id', 'price_timestamp', 'avg_mid_price']
    df1 = df[(df['item_id']==id1)][cols].sort_values('price_timestamp', ascending=True)
    df2 = df[(df['item_id']==id2)][cols].sort_values('price_timestamp', ascending=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = plt.figure(figsize=(6, 3))
        plt.plot(df1['price_timestamp'], df1[['avg_mid_price']])
        plt.ylabel('Price', fontsize=font_size)
        plt.xlabel('Date', fontsize=font_size)
        plt.gca().ticklabel_format(axis='y', style='plain')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with c2:
        fig = plt.figure(figsize=(6, 3))
        plt.plot(df2['price_timestamp'], df2[['avg_mid_price']])
        plt.ylabel('Price', fontsize=font_size)
        plt.xlabel('Date', fontsize=font_size)
        plt.gca().ticklabel_format(axis='y', style='plain')
        plt.xticks(rotation=45)
        st.pyplot(fig)

@st.cache_data
def find_correlation(df, s, threshold):
    new_df = df[['price_timestamp']].copy().drop_duplicates()
    for uid in s['item_id'].drop_duplicates().to_list():
        item_df = df[(df['item_id']==uid)][df.columns.to_list()]
        item_df.sort_values('price_timestamp', ascending=True, inplace=True)
        item_df = item_df[['price_timestamp', 'item_name', 'avg_mid_price']]
        item_df.rename(columns={'avg_mid_price':f'item_{uid}'}, inplace=True)
        new_df = pd.merge(new_df,item_df[['price_timestamp',f'item_{uid}']],'left',on='price_timestamp')
    new_df.sort_values('price_timestamp', inplace=True)
    new_df.drop('price_timestamp', axis=1, inplace=True)
    df_array = new_df.to_numpy()
    corr_matrix = np.corrcoef(df_array, rowvar=False)
    abs_corr = np.abs(corr_matrix)
    correlated_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(1+i, len(corr_matrix)):
            if abs_corr[i, j] > threshold:
                correlated_pairs.append((new_df.columns[i], new_df.columns[j], corr_matrix[i, j]))
    return correlated_pairs

def correlation_page(df, s):
    st.write(' ')
    c1, c2 = st.columns(2)
    num1 = c1.number_input('Enter an item ID:', step=1, key='num1')
    num2 = c2.number_input('Enter an item ID:', step=1, key='num2')
    if num1 and num2:
        draw_correlation(df, num1, num2)
    threshold = st.slider('Correlation Threshold', 0.8, 1.0, step=0.05, value=0.95)
    if st.checkbox('Find correlation'):
        correlated_pairs = find_correlation(df, s, threshold)
        if correlated_pairs:
            st.subheader(f'Items with abs correlation {threshold:.2f} or more:')
            for pair in correlated_pairs:
                st.write(f'{pair[0]} and {pair[1]} with correlation {pair[2]:.2f}')
        else:
            st.write('No highly correlated items found above the threshold.')

def userportfolio_page(df):
    st.write(' ')

def top100_delta(df, uid):
    item_df = df[(df['item_id']==uid)][df.columns.to_list()]
    item_df.sort_values('price_timestamp', ascending=True, inplace=True)
    if len(item_df) >= 2:
        item_df = item_df[['price_timestamp', 'item_name', 'avg_mid_price', 'total_volume']].tail(2)
        item = {
            'item_name': item_df['item_name'].iloc[0],
            'delta': item_df['avg_mid_price'].iloc[1] - item_df['avg_mid_price'].iloc[0]
        }
        return pd.DataFrame.from_dict([item], orient='columns')
    else:
        return pd.DataFrame(columns=['item_name', 'delta'])

def top100_volume(df, uid):
    item_df = df[(df['item_id']==uid)][df.columns.to_list()]
    item_df.sort_values('price_timestamp', ascending=True, inplace=True)
    if len(item_df) >= 1:
        item_df = item_df[['price_timestamp', 'item_name', 'avg_mid_price', 'total_volume']].tail(1)
        item = {
            'item_name': item_df['item_name'].iloc[0],
            'total_volume': item_df['total_volume'].iloc[0]
        }
        return pd.DataFrame.from_dict([item], orient='columns')
    else:
        return pd.DataFrame(columns=['item_name', 'total_volume'])

def top100_expense(df, uid):
    item_df = df[(df['item_id']==uid)][df.columns.to_list()]
    item_df.sort_values('price_timestamp', ascending=True, inplace=True)
    if len(item_df) >= 1:
        item_df = item_df[['price_timestamp', 'item_name', 'avg_mid_price', 'total_volume']].tail(1)
        item = {
            'item_name': item_df['item_name'].iloc[0],
            'avg_mid_price': item_df['avg_mid_price'].iloc[0]
        }
        return pd.DataFrame.from_dict([item], orient='columns')
    else:
        return pd.DataFrame(columns=['item_name', 'avg_mid_price'])

@st.cache_data
def top100_c_delta(df, s, is_fall):
    df2 = pd.concat(top100_delta(df, uid) for uid in s['item_id'].drop_duplicates().to_list())
    df2.set_index('item_name', inplace=True)
    df2.sort_values('delta', ascending=is_fall, inplace=True)
    return df2.head(100)

@st.cache_data
def top100_c_vol(df, s):
    df2 = pd.concat(top100_volume(df, uid) for uid in s['item_id'].drop_duplicates().to_list())
    df2.set_index('item_name', inplace=True)
    df2.sort_values('total_volume', ascending=False, inplace=True)
    return df2.head(100)

@st.cache_data
def top100_c_expense(df, s):
    df2 = pd.concat(top100_expense(df, uid) for uid in s['item_id'].drop_duplicates().to_list())
    df2.set_index('item_name', inplace=True)
    df2.sort_values('avg_mid_price', ascending=False, inplace=True)
    return df2.head(100)

def top100_page(df,s):
    st.write(' ')
    choice = st.radio('Action:', options=['Top 100 Price Rises', 'Top 100 Price Falls', 'Top 100 by Volume', 'Top 100 by Expense'])
    match choice:
        case 'Top 100 Price Rises':
            st.dataframe(top100_c_delta(df, s, False))
        case 'Top 100 Price Falls':
            st.dataframe(top100_c_delta(df, s, True))
        case 'Top 100 by Volume':
            st.dataframe(top100_c_vol(df, s))
        case 'Top 100 by Expense':
            st.dataframe(top100_c_expense(df, s))

@st.cache_data
def raw_page_set_index_item_id(_df):
    return _df.set_index('item_id', inplace=False)

def raw_page(s,d):
    st.write(' ')
    selected_option = st.radio('Select Table', options=['Static Data', 'Dynamic Data', 'Names', 'ID', 'Custom SQL'])
    if selected_option == 'Static Data':
        st.dataframe(raw_page_set_index_item_id(s))
    elif selected_option == 'Dynamic Data':
        st.dataframe(raw_page_set_index_item_id(d))
    elif selected_option == 'Names':
        #st.dataframe(s[['item_id', 'item_name']])
        user_input = st.text_input('Enter a partial item name:')
        if user_input:
            filtered_df = s[s['item_name'].str.contains(user_input, case=False)]
            if filtered_df.empty:
                st.write('No item found matching your search.')
            else:
                st.write('Item name:', filtered_df.iloc[0]['item_name'])
                st.write('Item ID:', filtered_df.iloc[0]['item_id'])
                fdf = filtered_df[['item_name', 'item_id', 'examine']].set_index('item_id', inplace=False)
                st.dataframe(fdf)
    elif selected_option == 'ID':
        #st.dataframe(s[['item_id', 'item_name']])
        user_input = st.number_input('Enter an item ID:', step=1)
        if user_input:
            filtered_df = s[(s['item_id']==user_input)][['item_id', 'item_name']]
            if filtered_df.empty:
                st.write('No item found matching your search.')
            else:
                st.write('Item name:', filtered_df.iloc[0]['item_name'])
    elif selected_option == 'Custom SQL':
        username = st.text_input('SQL username')
        password = st.text_input('SQL password')
        sql_command = st.text_input('SQL command')
        try:
            conn = psql.connect(
                database = st.secrets['DATABASE_NAME']
                , host = st.secrets['DATABASE_HOST']
                , port = int(st.secrets['DATABASE_PORT'])
                , user = username
                , password = password
            )
            cur = conn.cursor()
            try:
                cur.execute(sql_command)
                _df_data = cur.fetchall()
                _df_cols = [desc[0] for desc in cur.description]
                _df_df = pd.DataFrame(_df_data, columns=_df_cols)
                st.dataframe(_df_df)
            except Exception as e2:
                st.write(e2)
            finally:
                conn.close()
        except:
            st.write('Invalid SQL credentials.')



s, d = pinging_database()
df = pd.merge(s,d,'inner',on='item_id')
if 'page' not in st.session_state:
    st.session_state['page'] = 'Featured Items'
my_title = st.empty()
plt.style.use('classic')


st.sidebar.title('Navigation')
pages = [
    'Featured Items'
    , 'Market Indices'
    , 'Query Item'
    , 'Correlation'
    #, 'Portfolio'
    , 'Top 100'
    , 'Raw Data'
]
st.session_state['page'] = st.sidebar.selectbox(' ', pages, label_visibility='collapsed')
my_title.title(f'Dashboard: {st.session_state["page"]}')
match st.session_state['page']:
    case 'Featured Items':
        home_page(df)
    case 'Market Indices':
        marketindex_page(df,s)
    case 'Query Item':
        queryitem_page(df, s)
    case 'Correlation':
        correlation_page(df,s)
    case 'Portfolio':
        userportfolio_page(df)
    case 'Top 100':
        top100_page(df,s)
    case 'Raw Data':
        raw_page(s,d)
st.sidebar.title(' ')
st.sidebar.title('About')
st.sidebar.write('This is a protype finance dashboard for Digital Futures\' Data Engineering, which showcases economic data on the video game Oldschool Runescape.')



