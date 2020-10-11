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

    zscores = zscores.set_index(cols_gics + cols_stats, append=True).stack()\
        .reset_index()

    zscores.rename(columns={'level_14': 'AGENCY', 0: 'VALUE'}, inplace=True)

    return ratings, zscores, cols_agency


# ----------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------

ratings, zscores, cols_agency = load_data()


st.title('Aggregate Confusion')

st.markdown(
    """
    This is a demo of a Streamlit app that shows the Uber pickups
    geographical distribution in New York City. Use the slider
    to pick a specific hour and look at how the charts change.
    [See source code](https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py)
    """)

st.header('Firm Specific Disagreement')

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

st.sidebar.title('Visualization Settings')

number_of_companies =\
    st.sidebar.slider(label='Maximum # of companies',
                            min_value=20,
                            max_value=500,
                            value=100)

number_of_companies = number_of_companies * len(cols_agency)

agency = st.sidebar.selectbox(label='Select an agency view or stay agnostic',
                              options= ['AGNOSTIC'] + zscores['AGENCY'].unique().tolist(),
                              index=0)

# if agency == 'AGNOSTIC':
#     order = st.sidebar.radio(label='Select your prefered sorting option. [High-to-low]',
#                              options=('Agreement', 'Disagreement'))

# gics_1 = st.sidebar.multiselect(label='SECTOR',
#                                 options=zscores['SECTOR'].unique().tolist())




# ----------------------------------------------------------------------------
# Format
# ----------------------------------------------------------------------------

# if order == 'Agreement':
#     distance_sort_asc = True

# if order == 'Disagreement':
#     distance_sort_asc = False

if agency != 'AGNOSTIC':
    distance_col = 'DIST_TO_MEDIAN_{}'.format(agency)
else:
    distance_col = 'MEDIAN_AVG_DIST'



# zscores.sort_values([distance_col, 'NAME'],
#                     ascending=[False, True],
#                     inplace=True)


zscores_bar = zscores[['NAME', distance_col]].drop_duplicates()\
    .sort_values(distance_col, ascending=False)
zscores_bar = zscores_bar.reset_index(drop=True).reset_index()\
    .rename(columns={'index': 'COMPANY'})

fig_bar = px.line(zscores_bar,
                  x='COMPANY',
                  y=distance_col)



st.plotly_chart(fig_bar)

fig = px.scatter(zscores.head(number_of_companies).sort_values('MEDIAN'),
                 x='VALUE',
                 y='NAME',
                 color='AGENCY')

fig.update_layout(autosize=True,
                  width=1200, 
                  height=2000,
                  #margin=dict(l=40, r=40, b=40, t=40)
                  )

st.plotly_chart(fig)




# #https://bsou.io/posts/color-gradients-with-python
# def hex_to_RGB(hex):
#   ''' "#FFFFFF" -> [255,255,255] '''
#   # Pass 16 to the integer function for change of base
#   return [int(hex[i:i+2], 16) for i in range(1,6,2)]


# def RGB_to_hex(RGB):
#   ''' [255,255,255] -> "#FFFFFF" '''
#   # Components need to be integers for hex to make sense
#   RGB = [int(x) for x in RGB]
#   return "#"+"".join(["0{0:x}".format(v) if v < 16 else
#             "{0:x}".format(v) for v in RGB])

# def color_dict(gradient):
#   ''' Takes in a list of RGB sub-lists and returns dictionary of
#     colors in RGB and hex form for use in a graphing function
#     defined later on '''
#   return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
#       "r":[RGB[0] for RGB in gradient],
#       "g":[RGB[1] for RGB in gradient],
#       "b":[RGB[2] for RGB in gradient]}


# def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
#   ''' returns a gradient list of (n) colors between
#     two hex colors. start_hex and finish_hex
#     should be the full six-digit color string,
#     inlcuding the number sign ("#FFFFFF") '''
#   # Starting and ending colors in RGB form
#   s = hex_to_RGB(start_hex)
#   f = hex_to_RGB(finish_hex)
#   # Initilize a list of the output colors with the starting color
#   RGB_list = [s]
#   # Calcuate a color at each evenly spaced value of t from 1 to n
#   for t in range(1, n):
#     # Interpolate RGB vector for color at the current value of t
#     curr_vector = [
#       int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
#       for j in range(3)
#     ]
#     # Add it to our list of output colors
#     RGB_list.append(curr_vector)

#   return color_dict(RGB_list)



# def make_colors(df, value, center):

#     df['RANK'] = df.groupby(['YEAR'])[value].rank(method='first')
#     # Creates bins for ...
#     # High is bad
#     df.loc[df[value] >= center, 'BIN'] = df[df[value] >= center].groupby(['YEAR'])['RANK'].transform(
#         lambda x: pd.qcut(x, 5, labels=range(5,10)))
#     # Low is good
#     df.loc[df[value] < center, 'BIN'] = df[df[value] < center].groupby(['YEAR'])['RANK'].transform(
#         lambda x: pd.qcut(x, 5, labels=range(0,5)))
#     # Start color is good
#     gradient_obj = linear_gradient('#26c929','#cc0000', n=10)

#     # Create map dictionaries. 
#     r_dict = dict(enumerate(gradient_obj['r']))
#     g_dict = dict(enumerate(gradient_obj['g']))    
#     b_dict = dict(enumerate(gradient_obj['b']))
    
#     df['R'] = df['BIN'].map(r_dict)
#     df['G'] = df['BIN'].map(g_dict)
#     df['B'] = df['BIN'].map(b_dict)
    


# # This text element lets the user know the data is loading.
# data_load_state = st.text('Loading data...')
# # Load 10,000 rows of data into the dataframe.
# data = load_data(150000)
# # Notify the user that the data was successfully loaded.

# data_load_state.text('Loading data...done!')
# #data = data[['LON','LAT', 'VERIFIED_EMISSIONS', 'INSTALLATION_NAME', 'YEAR']]

# make_colors(data, value='VERIFIED_EMISSIONS_PCT_CHANGE', center=0)



# metric = st.sidebar.selectbox(label='Data Point',
#                               options=['VERIFIED_EMISSIONS',
#                                        'ALLOCATION',
#                                        'ALLOCATION_RESERVE',
#                                        'ALLOCATION_TRANSITIONAL'],
#                               index=0)

# year = st.sidebar.slider(label='Year',
#                          min_value=2008,
#                          max_value=2018,
#                          value=2018,
#                          step=1)

# scale = st.sidebar.slider(label='Scale',
#                          min_value=1,
#                          max_value=100,
#                          value=1,
#                          step=1)

# lower_percentile, upper_percentile = st.sidebar.slider(label='Percentile',
#                                                        min_value=0,
#                                                        max_value=100,
#                                                        value=(0,100),
#                                                        step=1)



# # -----------------------------------------------------------------------------
# # Set filters. 
# # -----------------------------------------------------------------------------

# data_year = data[data['YEAR']==year]

# mask_perc = (
#             (data_year[metric] >=
#              np.percentile(data_year[metric], lower_percentile)) &
#             (data_year[metric] <=
#              np.percentile(data_year[metric], upper_percentile)))

# data_perc = data_year[mask_perc]

# filtered_data = data_perc
# filtered_data.sort_values(metric, ascending=False, inplace=True)



# # # If the user doesn't want to select which features to control, these will be used.
# # default_control_features = ['Young','Smiling','Male']
# # if st.sidebar.checkbox('Show advanced options'):
# #     # Randomly initialize feature values. 
# #     features = get_random_features(feature_names, seed)
# #     # Let the user pick which features to control with sliders.
# #     control_features = st.sidebar.multiselect( 'Control which features?',
# #         sorted(features), default_control_features)
# # else:
# #     features = get_random_features(feature_names, seed)
# #     # Don't let the user pick feature values to control.
# #     control_features = default_control_features

# # # Insert user-controlled values from sliders into the feature vector.
# # for feature in control_features:
# #     features[feature] = st.sidebar.slider(feature, 0, 100, 50, 5)

# # # Generate a new image from this feature vector (or retrieve it from the cache).
# # with session.as_default():
# #     image_out = generate_image(session, pg_gan_model, tl_gan_model,
# #             features, feature_names)


# st.subheader('Map of all Installations')

# midpoint = (np.average(filtered_data['LAT']),
#             np.average(filtered_data['LON']))


# col_lyer = pdk.Layer(
#     'ColumnLayer',
#     data = filtered_data,
#     get_position = ['LON', 'LAT'],
#     get_elevation = '{0}'.format(metric),
#     elevation_scale = 1 / scale,
#     radius = 5000,
#     get_fill_color = ['R', 'G', 'B', 255],
#     pickable = True,
#     auto_highlight = True,
#     opacity = .5
#     )

# st.write(pdk.Deck(
#     map_style='mapbox://styles/mapbox/light-v9',
#     initial_view_state={
#         'latitude': midpoint[0],
#         'longitude': midpoint[1],
#         'min_zoom': 2,
#         'max_zoom': 10,
#         'zoom': 3,
#         'pitch': 40.5,
#         'bearing': -27.36
#     },
#     layers = [col_lyer],
#     tooltip = col_tooltip
# ))

# # legend = """
# #                 <style>
# #                 .bdot {{
# #                 height: 15px;
# #                 width: 15px;
# #                 background-color: Blue;
# #                 border-radius: 50%;
# #                 display: inline-block;
# #                 }}
# #                 .gdot {{
# #                 height: 15px;
# #                 width: 15px;
# #                 background-color: #4DFF00;
# #                 border-radius: 50%;
# #                 display: inline-block;
# #                 }}
# #                 </style>
# #                 </head>
# #                 <body>
# #                 <div style="text-align:left">
# #                 <h3>Legend</h3>
# #                 <span class="bdot"></span>  {} - {}<br>
# #                 <span class="gdot"></span>  &#62;{} - {}
# #                 </div>
# #                 </body>
# #                 """.format(round(min_val), round((max_val - min_val) / 2), round((max_val - min_val) / 2), round(max_val))

# # st.markdown(legend, unsafe_allow_html=True)


# # if st.checkbox('Show raw data', False):
# #     st.subheader('Raw data by ... year minute')
# #     st.write(filtered_data)
    
    
    
# # # -----------------------------------------------------------------------------
# # # Define Layers
# # # -----------------------------------------------------------------------------
# # hex_layer = pdk.Layer(
# #     'HexagonLayer',
# #     data=data,
# #     get_position=['lon', 'lat'],
# #     radius=100,
# #     elevation_scale=4,
# #     elevation_range=[0, 1000],
# #     pickable=True,
# #     extruded=True,
# # )

# # sct_layer = pdk.Layer(
# #     'ScatterplotLayer',
# #     data=data,
# #     get_position=['lon', 'lat'],
# #     auto_highlight=True,
# #     get_radius=10000,          # Radius is given in meters
# #     get_fill_color=[180, 0, 200, 140],  # Set an RGBA value for fill
# #     pickable=True)