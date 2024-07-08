# DF_2024_wk12_capstone_DE
# 📈 Financial Asset Dashboard for Oldschool Runescape

### Link

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rl-dfde-capstone-merge.streamlit.app/)

### About the project

Motivation: There is an increasing demand for data to make informed decisions when it comes to managing money. Investment is a choice, but market participation is not. Every party is a market participant, including large organisations. This project aims to use a toy api (with toy data) to prototype how data can be delivered to stakeholders in a user friendly fashion.

Description: A dashboard project to allow users to explore economic data from [an api](https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices). Project is decomposed into two parts:

- A data pipeline running on aws ec2 cron job that

  - extracts from the api
  - transforms data for cleaning
  - loads data into an AWS RDS database

- A streamlit dashboard for retrieving the data, and displaying

  - price and volume of featured items
  - weighted midprice of market indices
  - price, volume, and delta-KDE of user queried items
  - a list of correlated items
  - top 100 charts of: price rises, price falls, most traded, most expensive
  - lookup tables for interchanging item names and IDs
  - custom SQL query results

### Data Diagram

![Data diagram](https://github.com/RichardLoProjects/DF_2024_wk12_capstone_DE/blob/main/data_diagram.png)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
