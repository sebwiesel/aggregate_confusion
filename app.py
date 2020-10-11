#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 21:40:43 2020

@author: sebastian
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ----------------------------------------------------------------------------
# Preparations
# ----------------------------------------------------------------------------

PATH = r'ESG_RATINGS.xlsx'

@st.cache(allow_output_mutation=True)
def load_data():
    # Read from excel.
    df = pd.read_excel(PATH, skiprows=2)
    df['GICS'] = df['GICS'].astype(str)
    df['GICS'] = df['GICS'].str[:8]
    df = df.set_index(['GICS', 'ISIN', 'NAME'])
    # Replace -1 values in rating with nan.
    df['IS'] = df['IS'].replace(-1, pd.np.nan)

    cols_agency = ['SA',
                   'RS',
                   'A4',
                   'MS',
                   'IS']

    cols_gics = ['SECTOR',
                 'INDUSTRY GROUP',
                 'INDUSTRY',
                 'SUB INDUSTRY']

    # Remove rows with NA ratings.
    ratings = df[cols_agency + cols_gics].dropna()
    # Remove Zero values.
    ratings = ratings[(ratings != 0).all(1)]

    zscores = ratings.copy()
    # Calculate zscores & replace rating values.
    zscores[cols_agency] = zscores[cols_agency].apply(stats.zscore)
    # Calculate median zscore of each company.
    zscores['MEDIAN'] = zscores[cols_agency].median(axis=1)
    # Average absolute distance to the median aka median average distance.

    # def calculate_avg_dist_to_median(row):
    #     return sum(row - row.median())/len(row)

    zscores['MEDIAN_AVG_DIST'] = zscores[cols_agency]\
        .apply(lambda row: sum(abs(row - row.median()))/len(row),
               axis=1)

    # Calculate the distance to the median for each company and every rating.
    for c in cols_agency:
        zscores['DIST_TO_MEDIAN_{}'.format(c)] =\
            zscores['MEDIAN'] - zscores[c]

    cols_stats = [c for c in zscores.columns if 'MEDIAN' in c]

    zscores_wide = zscores
    zscores_wide = zscores_wide.reset_index()

    zscores = zscores.set_index(cols_gics + cols_stats, append=True).stack()\
        .reset_index()

    zscores.rename(columns={'level_14': 'AGENCY', 0: 'VALUE'}, inplace=True)

    zscores_long = zscores

    names = zscores_wide['NAME'].unique().tolist()

    return ratings, zscores_long, zscores_wide, cols_agency, names

# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------

ratings, zscores_long, zscores_wide, cols_agency, names = load_data()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

st.sidebar.title('Visualization Settings')

agency = st.sidebar.selectbox(label='Select an agency view or stay agnostic',
                              options=['AGNOSTIC'] + cols_agency,
                              index=0)

company_range =\
    st.sidebar.slider(label='Range of companies for raw data output',
                            min_value=0,
                            max_value=zscores_wide.shape[0],
                            value=(0, 100))

companies = st.sidebar.multiselect(label='Select companies for detailed view.',
                                   options=names,
                                   default=['WORKDAY INC-CLASS A',
                                            'METRO INC/CN',
                                            'MICRON TECHNOLOGY INC',
                                            'WEC ENERGY GROUP INC'])


# ----------------------------------------------------------------------------
# Format
# ----------------------------------------------------------------------------

if agency != 'AGNOSTIC':
    distance_col = 'DIST_TO_MEDIAN_{}'.format(agency)
else:
    distance_col = 'MEDIAN_AVG_DIST'

zscores_wide.sort_values(distance_col, ascending=False, inplace=True)
zscores_wide_dist = zscores_wide.copy()
zscores_wide_dist = zscores_wide_dist.reset_index(drop=True).reset_index()
zscores_wide_dist = zscores_wide_dist.rename(columns={'index': 'COMPANY'})


zscores_wide_raw = zscores_wide_dist[company_range[0]:company_range[1]]

zscores_long_sel = zscores_long[zscores_long['NAME'].isin(companies)]

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------


st.title('Aggregate Confusion')

st.markdown(
    '''
    Sustainable investing is becoming mainstream. $30 trillion of assets
    worldwide rely in some way on Environmental, Social, and Governance (ESG)
    data.

    The diverences in ESG ratings from various providers has been characterize
    in the paper [Aggregate Confusion: The Divergence of ESG Ratings]
    (http://dx.doi.org/10.2139/ssrn.3438533) by Berg, Kölbel & Rigobon.    

    This app applies the same statistical ideas outlined in research paper
    in order to allow users an interactive deep dive into the question of how
    different agencies rate different companies.

    Different ratings are used, as well as a more recent and wider dataset.

    Follow the link and take a look at the source code.
    [See source code](https://github.com/sebwiesel/aggregate_confusion)
    ''')

st.header('Firm Specific Disagreement')



# ----------------------------------------------------------------------------
# Distribution
# ----------------------------------------------------------------------------
st.subheader('Distribution of Universe of Companies')

fig_bar = px.line(zscores_wide_dist,
                  x='COMPANY',
                  y=distance_col)

st.plotly_chart(fig_bar)

# ----------------------------------------------------------------------------
# Raw Data
# ----------------------------------------------------------------------------
st.subheader('Raw Data')

st.write(zscores_wide_raw)

# ----------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------

fig = px.scatter(zscores_long_sel,
                 x='VALUE',
                 y='NAME',
                 color='AGENCY')

fig.update_layout(autosize=True
                  #width=1200, 
                  #height=2000,
                  #margin=dict(l=40, r=40, b=40, t=40)
                  )

st.plotly_chart(fig)
