# DF_2024_wk12_capstone_DE
# 📊 Financial Asset Dashboard for Oldschool Runescape

A dashboard project to allow users to explore economic data from [an api](https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices). Project is decomposed into two parts:

1. A data pipeline that extracts from the api, transforms data for cleaning, and loads data into an AWS RDS database.

2. A streamlit dashboard for retrieving the data, and displaying

 - price and volume of featured items
 - weighted midprice of market indices
 - price, volume, and delta-KDE of user queried items
 - a list of correlated items
 - top 100 charts of: price rises, price falls, most traded, most expensive
 - lookup tables for interchanging item names and IDs
 - custom SQL query results


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rl-dfde-capstone-merge.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
