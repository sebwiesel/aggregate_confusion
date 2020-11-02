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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from correlation import correlation_plot, correlation_color_bar


# ----------------------------------------------------------------------------
# Preparations
# ----------------------------------------------------------------------------

PATH = r'ESG_RATINGS.xlsx'


@st.cache(allow_output_mutation=True)
def load_data(size):
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
    df.drop_duplicates(inplace=True, subset='NAME')

    df['GICS'] = df['GICS'].astype(str)
    df['GICS'] = df['GICS'].str[:8]
    df = df.set_index(['GICS', 'ISIN', 'NAME'])
    # Replace -1 values in rating with nan.
    df['IS'] = df['IS'].replace(-1, np.nan)

    cols_raters = ['SA',
                   'RS',
                   'A4',
                   'MS',
                   'IS']

    cols_gics = ['SECTOR',
                 'INDUSTRY GROUP',
                 'INDUSTRY',
                 'SUB INDUSTRY']

    # Remove rows with NA ratings.
    ratings = df[cols_raters + cols_gics].dropna()
    # Remove Zero values.
    ratings = ratings[(ratings != 0).all(1)]

    zscores = ratings.copy()
    # Calculate zscores & replace rating values.
    zscores[cols_raters] = zscores[cols_raters].apply(stats.zscore)
    # Calculate median zscore of each company.
    zscores['MEDIAN'] = zscores[cols_raters].median(axis=1)
    # Calculate mean zscore of each company.
    zscores['MEAN'] = zscores[cols_raters].mean(axis=1)

    # Mean Absolute Distance
    zscores['MEAN_ABS_DIST'] = zscores[cols_raters]\
        .apply(lambda row: sum(abs(row - row.mean()))/len(row),
               axis=1)

    # This metric was used in an earlier version of the paper.
    # Average absolute distance to the median aka median average distance.
    zscores['MEDIAN_ABS_DIST'] = zscores[cols_raters]\
        .apply(lambda row: sum(abs(row - row.median()))/len(row),
               axis=1)

    # Calculate the distance to the median for each company and every rating.
    for c in cols_raters:
        zscores['DIST_TO_MEDIAN_{}'.format(c)] =\
            zscores[c] - zscores['MEDIAN']

    # Calculate the distance to the mean for each company and every rating.
    for c in cols_raters:
        zscores['DIST_TO_MEAN_{}'.format(c)] =\
            zscores[c] - zscores['MEAN']

    # Calculate percentile column for each rater.
    for c in cols_raters:
        zscores['PCTL_{}'.format(c)] = zscores[c].rank(pct=True) * 100

    # List with all columns containing 'MEDIAN' or 'PCTL'.
    cols_stats = [c for c in zscores.columns if 'MEDIAN' in c]
    cols_stats = cols_stats + [c for c in zscores.columns if 'PCTL' in c]
    cols_stats = cols_stats + [c for c in zscores.columns if 'MEAN' in c]
    # The preferable one-liner causes error in streamlit. ???
    # cols_stats =[c for c in zscores.columns if any(w in c for w in ['MEDIAN', 'MEAN', 'PCTL'])]

    zscores_wide = zscores
    zscores_wide = zscores_wide.reset_index()

    zscores = zscores.set_index(cols_gics + cols_stats, append=True).stack()\
        .reset_index()


    # First convert all column names to string.
    cols = [str(c) for c in list(zscores)]
    col_to_rename = [c for c in cols if 'level_' in c][0]
    # Rename one column dynamic.
    zscores.rename(columns={col_to_rename: 'RATER', 0: 'VALUE'}, inplace=True)

    zscores_long = zscores

    names = zscores_wide['NAME'].unique().tolist()

    # Select companies at random.
    default_names = zscores_wide.sample(size)['NAME']

    return ratings, zscores_long, zscores_wide, cols_raters, names, default_names

# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------


# Number of companies in 'Ratings' scatter plot.
size = 25

ratings, zscores_long, zscores_wide, cols_raters, names, default_names = load_data(size)
# Count for highlevel info display.
count_company = zscores_wide.shape[0]

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

# Add a horizontal separator.
st.sidebar.markdown('<hr size="4">', unsafe_allow_html=True)
st.sidebar.subheader(':world_map: Navigate')
# Add the radio button with default set to 'Correlation'
page = st.sidebar.radio(label='', options=pages, index=1)
# Add a horizontal separator.
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
    st.warning(''''
               :construction: **Under Construction** :construction:

               Sorry for the dust! We know it’s taking a while but sit tight
               and we’ll be with you soon.
               ''')

# ----------------------------------------------------------------------------
# Page - Correlation
# ----------------------------------------------------------------------------

if page == 'Correlation':

    st.markdown('''
                The table below shows the Pearson correlations between the
                aggregate ESG ratings of each rater.
               ''')
               
    #... and The highest level of
    #agreement between each other, with a correlation of 0.71. 
    #rall correlations, with an average of 0.53. The social dimension is on
    #average correlated at 0.42, and the governance dimension has the lowest
    #correlation, with an average of 0.30. KLD and MSCI clearly exhibit the
    #lowest correlations with other raters, both for the rating and for the
    #individual dimensions. These results are largely consistent with prior 
    #findings by Chatterji et al. (2016).
    # -------------------------------------------------------------------------
    # ....
    # -------------------------------------------------------------------------

    g = correlation_plot(zscores_wide[cols_raters])

    # #sns.set_style("whitegrid", {'axes.grid' : False})
    # sns.set_style("whitegrid")

    # df = zscores_wide.loc[:, cols_raters]
    
    # df.columns = df.columns.map("-".join)
    
    # # Compute a correlation matrix and convert to long-form
    # corr_mat = df.corr().stack().reset_index(name="correlation")
    # corr_mat['correlation'] = corr_mat['correlation'] * 100
    # # Draw each cell as a scatter point with varying size and color
    # g = sns.relplot(
    #     data=corr_mat,
    #     x="level_0", y="level_1", hue="correlation", size="correlation",
    #     palette="vlag", hue_norm=(-100, 100), edgecolor=".7",
    #     height=10, sizes=(50, 250), size_norm=(-.2, .8),
    # )
    
    # # Tweak the figure to finalize
    # g.set(xlabel="", ylabel="", aspect="equal")
    # g.despine(left=True, bottom=True)
    # g.ax.margins(.02)
    # for label in g.ax.get_xticklabels():
    #     label.set_rotation(90)
    # for artist in g.legend.legendHandles:
    #     artist.set_edgecolor(".7")    
    st.pyplot(g)
    
    f = correlation_color_bar()
    st.pyplot(f)
    
    
    # -------------------------------------------------------------------------
    # ....
    # -------------------------------------------------------------------------

    # fig, ax = plt.subplots(figsize=(12, 10))
    # # mask
    # mask = np.triu(np.ones_like(zscores_wide[cols_raters].corr(), dtype=np.bool))
    # # adjust mask and df
    # mask = mask[1:, :-1]
    # corr = zscores_wide[cols_raters].corr().iloc[1:,:-1].copy()
    # # color map
    # cmap = sb.diverging_palette(0, 230, 90, 60, as_cmap=True)
    # # plot heatmap
    # sb.heatmap(corr, mask=mask, annot=True, fmt=".2f", 
    #            linewidths=5, cmap=cmap, vmin=-1, vmax=1, 
    #            cbar_kws={"shrink": .8}, square=True)
    # # ticks
    # yticks = [i.upper() for i in corr.index]
    # xticks = [i.upper() for i in corr.columns]
    # plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
    # plt.xticks(plt.xticks()[0], labels=xticks)

    # st.header(':mag_right: Investigate Disagreement & Agreement in Ratings')
    # st.pyplot(fig)


# ----------------------------------------------------------------------------
# Page - Ratings
# ----------------------------------------------------------------------------

if page == 'Ratings':

    st.markdown(
        '''
        This section allows you to take a look at how ratings between different
        ESG providers differ. Below you can select a specific ESG rater's view
        or stay agnostic.
        ''')

    rater = st.selectbox(
        label="Select a specific ESG rater's view or stay agnostic",
        options=['AGNOSTIC'] + cols_raters,
        index=0)

    col_left, _,  col_right = st.beta_columns([10, 1, 10])

    if rater != 'AGNOSTIC':
        distance_col = 'DIST_TO_MEAN_{}'.format(rater)
        distance_stat = 'Distance to Mean'

        st.markdown(
            '''
            You are taking {}'s view. For each of the {} companies in our
            universe the chart below displays the **{} Rating Distance to the
            Mean** ({}-DM). This measure indicates if the rater is above or
            below consensus and by which margin.
            '''.format(rater, count_company, rater, rater))

        with col_left:
            st.latex(
                r'''
                \begin{aligned}
                \\
                \Large \textrm{DM}_{%s} = x_{%s}-\overline{x}
                \end{aligned}
                ''' % (rater.lower(), rater.lower()))

        with col_right:
            st.latex(
                r'''
                \begin{aligned}
                \textrm{where:}\\
                x_{%s}&=\textrm{i-th data value in the set}\\
                \overline{x}&=\textrm{average value of the data set}
                \end{aligned}
                ''' % rater.lower()
                )   


    # Stay Agnostic
    else:
        distance_col = 'MEAN_ABS_DIST'
        distance_stat = 'Mean Absolute Deviation'

        st.markdown(
            '''
            For each of the {} companies in our universe the chart below displays
            an overall measure of agreement / disagreement among ESG raters.
            This measure is the **Mean Absolute Deviation** (MAD).
            '''.format(count_company))

        with col_left:
            st.write('')
            st.latex(
                r'''
                \Large \textrm{MAD} = \frac{1}{n} \sum_{i=1}^{n} |x_i-\overline{x}|
                ''')

        with col_right:

            st.latex(
                r'''
                \begin{aligned}
                \textrm{where:}\\
                x_i&=\textrm{i-th data value in the set}\\
                n&=\textrm{number of data values}\\
                \overline{x}&=\textrm{average value of the data set}
                \end{aligned}
                ''')

    zscores_wide.sort_values(distance_col, ascending=False, inplace=True)
    zscores_wide_dist = zscores_wide.copy()
    zscores_wide_dist = zscores_wide_dist.reset_index(drop=True).reset_index()
    zscores_wide_dist = zscores_wide_dist.rename(columns={'index': 'COMPANY'})

    # ------------------------------------------------------------------------
    # Distribution - Plot
    # ------------------------------------------------------------------------
    st.header(':round_pushpin: Histogram of {}'.format(distance_stat))

    hist = px.histogram(zscores_wide,
                        x=distance_col,
                        #y='NAME',
                        marginal='box',
                        # hover_data=df.columns
                        )

    # hist.update_layout(shapes=[
    #     dict(
    #       type= 'line',
    #       yref= 'paper', y0= 0, y1= 1,
    #       xref= 'x', x0= 5, x1= 5
    #     )
    # ])

    hist.update_layout(legend=dict(orientation="h",
                                   yanchor='bottom',
                                   y=1.02,
                                   xanchor='right',
                                   x=1),
                       margin=dict(t=0,
                                   r=0,
                                   b=0,
                                   l=0),
                       bargap = 0.3,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0, 0, 0, 0)')

    st.plotly_chart(hist)
  
    company_range =\
    st.slider(label='Range of companies for raw data output',
              min_value=0,
              max_value=zscores_wide.shape[0],
              value=(0, 100))

    # ------------------------------------------------------------------------
    # Selection - Scatterplot
    # ------------------------------------------------------------------------

    st.header(':mag_right: Investigate Disagreement & Agreement in Ratings')

    # TODO: figure out how to target a specific radio button element.
    #st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    selection = st.radio(label='', options=['Disagreement',
                                            'Agreement',
                                            'Custom'])

    if selection == 'Custom':

        companies = st.multiselect(label='Select companies for detailed view.',
                                   options=names,
                                   default=default_names)
        zscores_wide_sel = zscores_wide[zscores_wide['NAME'].isin(companies)]

    if selection == 'Disagreement':
        zscores_wide_sel = \
            zscores_wide.sort_values(['MEAN_ABS_DIST'],
                                     ascending=False).head(size)

    if selection == 'Agreement':
        zscores_wide_sel = \
            zscores_wide.sort_values(['MEAN_ABS_DIST']).head(size)

    # Always sort data by MEAN.
    zscores_wide_sel = zscores_wide_sel.sort_values(['MEAN'])

    fig_sel = make_subplots(rows=1,
                            cols=2,
                            shared_yaxes=True,
                            column_widths=[0.7, 0.3],
                            vertical_spacing=0.05,
                            horizontal_spacing=0.05)

    # Start with a plotting the mean of each observation as line chart.
    fig_sel.add_trace(go.Scatter(x=zscores_wide_sel['MEAN'],
                                 y=zscores_wide_sel['NAME'],
                                 name='Mean',
                                 line=dict(color='royalblue',
                                           width=1.5,
                                           dash='dot',
                                           shape='spline'),
                                 text=["tweak line smoothness<br>with 'smoothing' in line object"],
                                 #hoverinfo='text+name',
                                 ),
                      row=1, col=1)
    # 
    # Sustainalitics Gold
    for e in cols_raters:
        fig_sel.add_trace(go.Scatter(x=zscores_wide_sel[e],
                                     y=zscores_wide_sel['NAME'],
                                     name=e,
                                     mode='markers',
                                     opacity=0.75,
                                     marker=dict(
                                         size=11,
                                         line=dict(color='Black',
                                                   width=1))),
                          row=1, col=1)

    # ------------------------------------------------------------------------
    # Lolipop Chart
    # ------------------------------------------------------------------------

    # Offset the line length by the marker size to avoid overlapping
    marker_offset = 0.06

    def offset_signal(signal, marker_offset):
        if abs(signal) <= marker_offset:
            return 0
        return signal - marker_offset if signal > 0 else signal + marker_offset

    # Dots for lolipop
    fig_sel.add_trace(go.Scatter(x=zscores_wide_sel[distance_col],
                                 y=zscores_wide_sel['NAME'],
                                 name='MAD',
                                 mode='markers',
                                 opacity=1,
                                 marker=dict(
                                     color='red',
                                     size=11,
                                     line=dict(color='Black',
                                               width=1))),
                      row=1, col=2)

    # Add shapes for lolipop.    
    fig_sel.update_layout(
        shapes=[dict(type='line',
                     xref='x2',
                     yref='y2',
                     x0=0,
                     y0=i,
                     x1=offset_signal(list(zscores_wide_sel[distance_col])[i], marker_offset),
                     y1=i,
                     line=dict(
                        color='grey',
                        width=3
                    )
                ) for i in range(len(zscores_wide_sel['NAME']))],)

    # ------------------------------------------------------------------------
    # Style Axis & Ranges
    # ------------------------------------------------------------------------

    # Update ranges.
    fig_sel.update_layout(xaxis=dict(range=[-2, 2]),
                          # Lolipop x axis
                          xaxis2=dict(range=[zscores_wide[distance_col].min() - marker_offset,
                                             zscores_wide[distance_col].max() + marker_offset]),
                          yaxis_type='category',
                          yaxis=dict(range=[-.6,
                                            zscores_wide_sel.shape[0] - .5]),)


    # Vertical axes. Lolipop
    fig_sel.update_xaxes(showgrid=False,
                         showline=True,
                         linewidth=2,
                         linecolor='black',
                         zeroline=True,
                         zerolinewidth=1.5,
                         zerolinecolor='silver',
                         row=1,
                         col=2)

    fig_sel.update_yaxes(showgrid=False,
                         #gridcolor='silver',
                         showline=True,
                         linewidth=2,
                         linecolor='black',
                         row=1,
                         col=2)  


    # Vertical axes. Scatter
    fig_sel.update_xaxes(showline=True,
                         linewidth=2,
                         linecolor='black',
                         row=1,
                         col=1)
    
    fig_sel.update_yaxes(showgrid=True,
                         gridcolor='silver',
                         gridwidth=1,
                         #gridcolor='silver',
                         showline=True,
                         linewidth=2,
                         linecolor='black',
                         #zeroline=True,
                         #zerolinewidth=2,
                         #zerolinecolor='LightPink',
                         row=1,
                         col=1)  
  
          

    # ------------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------------

    fig_sel.update_layout(legend=dict(orientation="h",
                                      yanchor='bottom',
                                      y=1.02,
                                      xanchor='right',
                                      x=1),
                          height=1000,
                          margin=dict(t=100,
                                      r=0,
                                      b=0,
                                      l=0),
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    

    
                      
                                # yaxis={'title': '',
                                #         'visible': True,
                                #         'showticklabels': True})

    # fig_sel.update_layout(autosize=True

    #                   #margin=dict(l=40, r=40, b=40, t=40)
    #                   )

    st.plotly_chart(fig_sel)
    



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
        st.subheader(':rotating_light: Bottom Companies')
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

