#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 21:40:43 2020

@author: sebastian
"""

import streamlit as st
import pandas as pd
from scipy import stats
import plotly.express as px
import numpy as np
import matplotlib

# ----------------------------------------------------------------------------
# Preparations
# ----------------------------------------------------------------------------

PATH = r'ESG_RATINGS.xlsx'


@st.cache(allow_output_mutation=True)
def load_data():
    # Read from excel.
    df = pd.read_excel(PATH, skiprows=2)
    df.dropna(how='all', inplace=True)

    max_name = 19

    def cut_names(name, max_name):
        while len(name) > max_name:
            words = name.split(' ')
            word_count = len(words)
            if word_count == 1:
                name = name[:max_name]
            else:
                name = ' '.join(words[:-1])
        return name

    df['NAME'] = df['NAME'].apply(lambda x: cut_names(x, max_name))

    df['GICS'] = df['GICS'].astype(str)
    df['GICS'] = df['GICS'].str[:8]
    df = df.set_index(['GICS', 'ISIN', 'NAME'])
    # Replace -1 values in rating with nan.
    df['IS'] = df['IS'].replace(-1, np.nan)

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
    zscores['MEDIAN_AVG_DIST'] = zscores[cols_agency]\
        .apply(lambda row: sum(abs(row - row.median()))/len(row),
               axis=1)

    # Calculate the distance to the median for each company and every rating.
    for c in cols_agency:
        zscores['DIST_TO_MEDIAN_{}'.format(c)] =\
            zscores['MEDIAN'] - zscores[c]

    # Calculate percentile column for each agency.
    for c in cols_agency:
        zscores['PCTL_{}'.format(c)] = zscores[c].rank(pct=True) * 100

    # List with all columns containing 'MEDIAN' or 'PCTL'.
    cols_stats = [c for c in zscores.columns if 'MEDIAN' in c]
    cols_stats = cols_stats + [c for c in zscores.columns if 'PCTL' in c]
    # The preferable one-liner causes error in streamlit. ???
    # cols_stats =[c for c in zscores.columns if any(w in c for w in ['MEDIAN', 'PCTL'])]

    zscores_wide = zscores
    zscores_wide = zscores_wide.reset_index()

    zscores = zscores.set_index(cols_gics + cols_stats, append=True).stack()\
        .reset_index()

    # TODO: add dynamic level count.
    zscores.rename(columns={'level_19': 'AGENCY', 0: 'VALUE'}, inplace=True)

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

pages = ['Home', 'Correlation', 'Ratings', 'Rankings']

st.sidebar.title('Aggregate Confusion')

st.sidebar.markdown(
    '''
    Sustainable investing is becoming mainstream. $30 trillion of assets
    globaly rely on Environmental, Social, and Governance (ESG)
    data.
    
    **This app allows users an interactive deep dive into the question of how
    the ratings of different ESG providers differ.**
    ''')

st.sidebar.markdown('<hr size="4">', unsafe_allow_html=True)
st.sidebar.subheader(':world_map: Navigate')
page = st.sidebar.radio(label='', options=pages, index=3)
st.sidebar.markdown('<hr size="4">', unsafe_allow_html=True)

st.sidebar.markdown(
    '''
    Follow the link and take a look at the source code.
    [See source code](https://github.com/sebwiesel/aggregate_confusion)
    ''')

# ----------------------------------------------------------------------------
# Page - Home
# ----------------------------------------------------------------------------

st.title(page)
if page == 'Home':
    st.warning('''':construction: **Under Construction** :construction:
               
               Sorry for the dust! We know it’s taking a while but sit tight
               and we’ll be with you soon.
               ''')


if page == 'Correlation':
    st.warning('''':construction: **Under Construction** :construction:
               
               Sorry for the dust! We know it’s taking a while but sit tight
               and we’ll be with you soon.
               ''')    

if page == 'Ratings':

    rater = st.selectbox(label='Select an agency view or stay agnostic',
                         options=['AGNOSTIC'] + cols_agency,
                         index=0)
    
    if rater != 'AGNOSTIC':
        distance_col = 'DIST_TO_MEDIAN_{}'.format(agency)
    else:
        distance_col = 'MEDIAN_AVG_DIST'

    zscores_wide.sort_values(distance_col, ascending=False, inplace=True)
    zscores_wide_dist = zscores_wide.copy()
    zscores_wide_dist = zscores_wide_dist.reset_index(drop=True).reset_index()
    zscores_wide_dist = zscores_wide_dist.rename(columns={'index': 'COMPANY'})
    
    # -------------------------------------------------------------------------
    # Distribution
    # -------------------------------------------------------------------------
    st.subheader('Distribution of Universe of Companies')

    fig_bar = px.line(zscores_wide_dist,
                      x='COMPANY',
                      y=distance_col)

    st.plotly_chart(fig_bar)
    
    company_range =\
    st.sidebar.slider(label='Range of companies for raw data output',
                            min_value=0,
                            max_value=zscores_wide.shape[0],
                            value=(0, 100))

    zscores_wide_raw = zscores_wide_dist[company_range[0]:company_range[1]]

    st.header('Disagreement in Ratings')

    

    
    # ----------------------------------------------------------------------------
    # Raw Data
    # ----------------------------------------------------------------------------
    st.subheader('Raw Data')

    st.write(zscores_wide_raw)

    # ----------------------------------------------------------------------------
    # Selection
    # ----------------------------------------------------------------------------
    companies = st.sidebar.multiselect(label='Select companies for detailed view.',
                               options=names,
                               default=['WORKDAY INC-CLASS A',
                                        'METRO INC/CN',
                                        'MICRON TECHNOLOGY INC',
                                        'WEC ENERGY GROUP INC'])
    
    zscores_long_sel = zscores_long[zscores_long['NAME'].isin(companies)]
    
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

# ----------------------------------------------------------------------------
# Page - Rankings
# ----------------------------------------------------------------------------

if page == 'Rankings':
    
    st.markdown(
        '''
        Rankings can be more important than the individual score in many
        financial applications. Investors often want to construct a portfolio
        with sustainability leaders from the top quantile, or alternatively
        exclude sustainability laggards from the bottom quantile. With this
        approach, the disagreement in ratings would be less relevant than the
        disagreement in rankings.
        ''')    


    # ------------------------------------------------------------------------
    # Top & Bottom Companies by Ranking
    # ------------------------------------------------------------------------
    
    st.header('Top & Bottom Companies by Ranking')

    # container = st.beta_container()

    top_bottom_quantile =\
        st.slider(label='Select a quantile :)',
                  min_value=0,
                  max_value=50,
                  value=20)

    # ------------------------------------------------------------------------
    # Format
    # ------------------------------------------------------------------------
    ranking_wide = zscores_wide.copy()
    col_pctl = [c for c in ranking_wide.columns if 'PCTL' in c]

    # Calculate average percentile ranking.
    ranking_wide['PCTL_AVG'] = ranking_wide[col_pctl].mean(axis=1)

    # Sort the selected companies by their average ranking descending.
    ranking_wide.sort_values(['PCTL_AVG'], ascending=False, inplace=True)

    # Select the companies that are in the top & bottom quantile across all
    # raters.
    mask_top = ranking_wide[col_pctl] >= 100 - top_bottom_quantile
    mask_bottom = ranking_wide[col_pctl] <= top_bottom_quantile

    # Subset for the identified rows.
    ranking_wide_top = ranking_wide[mask_top.all(axis=1)]
    ranking_wide_bottom = ranking_wide[mask_bottom.all(axis=1)]

    # Counts for highlevel info display.
    count_company = ranking_wide.shape[0]
    count_top = ranking_wide_top.shape[0]
    count_bottom = ranking_wide_bottom.shape[0]
    
    # ...
    perfect = count_company * top_bottom_quantile/100

    # Reset index to get consecutive index numbers for output.
    # Start from 1 to n. !!! The frames are already sorted !!!
    ranking_wide_top.index = np.arange(1, count_top + 1)

    # Quick resort for bottom rankings.
    ranking_wide_bottom.sort_values(['PCTL_AVG'], ascending=True, inplace=True)
    # Start from 1 to n. !!! The frames are already sorted !!!
    ranking_wide_bottom.index = np.arange(1, count_bottom + 1)

    # Define final display columns.
    col_out = ['NAME', 'PCTL_AVG']

    st.markdown(
        '''
        Take a look at the companies that are in the top and bottom {}%
        bucket across **all** 5 ESG raters based on the **ranked**
        scores. For reference, if there is perfect agreement between the 5 ESG raters
        each bucket would contain {:,.0f} companies.
        '''.format(top_bottom_quantile, perfect))


    col_top, _,  col_bottom = st.beta_columns([10, 1, 10])

    with col_top:
        st.subheader(':trophy: Top Companies')
        st.markdown(
            '''
            <span style="color:green">*{}*</span>
            out of {} firms are in in the top {}% across all ESG raters.
            '''.format(count_top, count_company, top_bottom_quantile),
            unsafe_allow_html=True)

        st.table(ranking_wide_top[col_out].style\
                 .format({'PCTL_AVG': '{:,.2f}'})\
                 .background_gradient(cmap='Greens', subset=['PCTL_AVG',])
                 #.bar(subset=['PCTL_AVG',], color='lightgreen')
                 )
        

    with col_bottom:
        st.subheader(':thumbsdown: Bottom Companies')
        st.markdown(
            '''
            <span style="color:red">*{}*</span>
            out of {} firms are in in the bottom {}% across all ESG raters.
            '''.format(count_bottom, count_company, top_bottom_quantile),
            unsafe_allow_html=True)
    
        st.table(ranking_wide_bottom[col_out].style\
                 .format({'PCTL_AVG': '{:,.2f}'})\
                 .background_gradient(cmap=matplotlib.cm.get_cmap('Reds_r'), subset=['PCTL_AVG',])
                 #.bar(subset=['PCTL_AVG',], color='lightgreen')
                 )


    # Sustainable investing is becoming mainstream. $30 trillion of assets
    # globaly rely on Environmental, Social, and Governance (ESG)
    # data.
    
    
    
    # In [Aggregate Confusion: The Divergence of ESG Ratings]
    # (http://dx.doi.org/10.2139/ssrn.3438533) by Berg, Kölbel & Rigobon the
    # authors.    

    # This app applies the same statistical ideas outlined in research paper
    # - and adds a few original twists - in order to allow users an interactive
    # deep dive into the question of how different agencies rate different
    # companies.

    # Different ratings are used, as well as a more recent and wider dataset.

    # Follow the link and take a look at the source code.
    #[See source code](https://github.com/sebwiesel/aggregate_confusion)

