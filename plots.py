import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def create_box_plot(df, x, y):
    """
    INPUT:
        df: pandas DataFrame - dataframe for the plot
        x: str - name of column for x axis
        y: str - name of column for y axis

    OUTPUT:
        fig2: plotly figure

    Description:
        The function creates customized box plot
    """

    fig2 = px.box(df, x=x, y=y, points=False, height=400)

    fig2.update_traces(
                    marker_line_width=1.,
                    marker_color='#A41A50',
                        )
    fig2.update_xaxes(title = dict(text="<b>City</b>", font_family='Courier New',font_size=14),
                     tickfont = dict(size=10, family='Courier New',)
                    )
    fig2.update_yaxes(title = dict(text="<b>Price,USD</b>", font_family='Courier New',font_size=14),
                     showgrid=True,
                     tickfont = dict(size=10, family='Courier New',),
                    )
    fig2.update_layout(
                        template = 'simple_white',
                        title=dict(text="<b>Distribution of price by city</b>", font_family='Courier New',font_size=16),

                        plot_bgcolor='#F0F1D4',
                        paper_bgcolor= '#F3EEF0'
                        )

    fig2.show()
    return fig2


def create_bar_subplots(dfs, x_column, y_column, min, max, title):
    """
    INPUT:
        dfs: list of pandas DataFrame - dataframes for the plots
        x_column: str - name of column for x axis
        y_column: str - name of column for y axis
        min: int - min value of dataframe slice
        max: int - max value of dataframe slice
        title: str - plot title

    OUTPUT:
        fig1: plotly figure

    Description:
        The function creates customized bar plots
    """

    fig1 = make_subplots(rows=1, cols=5, subplot_titles=(dfs[0].iloc[0,0], dfs[1].iloc[0,0], dfs[2].iloc[0,0], dfs[3].iloc[0,0], dfs[4].iloc[0,0]), shared_yaxes=True)

    fig1.add_trace(
        go.Bar(x=dfs[0][min:max][x_column], y=dfs[0][min:max][y_column], text=dfs[0][min:max][y_column]),
        row=1, col=1
    )
    fig1.add_trace(
        go.Bar(x=dfs[1][min:max][x_column], y=dfs[1][min:max][y_column], text=dfs[1][min:max][y_column]),
        row=1, col=2
    )
    fig1.add_trace(
        go.Bar(x=dfs[2][min:max][x_column], y=dfs[2][min:max][y_column], text=dfs[2][min:max][y_column]),
        row=1, col=3
    )
    fig1.add_trace(
        go.Bar(x=dfs[3][min:max][x_column], y=dfs[3][min:max][y_column], text=dfs[3][min:max][y_column]),
        row=1, col=4
    )
    fig1.add_trace(
        go.Bar(x=dfs[4][min:max][x_column], y=dfs[4][min:max][y_column], text=dfs[4][min:max][y_column]),
        row=1, col=5
    )
    fig1.update_traces(
                    marker_line_width=2.,
                    marker_color='#38D7A0',
                    marker_line_color='black',
                    texttemplate='%{text:.0f}', textposition='inside',
                    insidetextfont = {'color' : 'black', 'size' : 12},
                    hovertemplate =
                    '<i>Mean price</i>: $%{y:.0f}'+
                    '<extra></extra>'
                        )
    fig1.update_xaxes(tickfont = dict(size=6, family='Courier New'), tickangle=45,
                    )
    fig1.update_xaxes(title = dict(text="<b>Neighbourhood</b>", font_family='Courier New',font_size=16),
                     tickfont = dict(size=8, family='Courier New',), row=1, col=3
                    )
    fig1.update_yaxes(showgrid=True)



    fig1.update_yaxes(
        title = dict(text="<b>Price,USD</b>", font_family='Courier New',font_size=14),
                     showgrid=True,
                     tickfont = dict(size=12, family='Courier New',),row=1,col=1
                    )
    fig1.update_layout(
                        template = 'simple_white',
                        title=dict(text=title, font_family='Courier New',font_size=15),
                        margin=dict(l=20, r=20, t=50, b=20),
                        plot_bgcolor='#F0F1D4',
                        paper_bgcolor= '#F3EEF0',
                        height=600,
                        showlegend=False,
                        yaxis_range=[0,350]
                        )
    fig1.for_each_annotation(lambda a: a.update(text=f'<b>{a.text}</b>'))
    fig1.update_annotations(font=dict(family='Courier New', size=12))
    fig1.show()
    return fig1


def create_pretty_table(df, list_of_tuple):
    """
    INPUT:
        df: pandas DataFrame - table to be styled
        list_of_tuple: list of tuples - table has multiindex columns and can be accessed with tuple

    OUTPUT:
        pretty_table: padnas DataFrame - styled pandas DataFrame

    Description:
        The function creates styling of a table, specifically background gradient
    """
    styles = [
    dict(selector="th", props=[("font-size", "100%"),
                               ("text-align", "center")])
    ]
    pretty_table = df.reset_index(drop=True)\
    .style.background_gradient(cmap=sns.light_palette("skyblue", as_cmap=True), \
     subset=[list_of_tuple[1], list_of_tuple[3], list_of_tuple[5], list_of_tuple[7], list_of_tuple[9]]).hide_index()\
    .set_precision(2).set_table_styles(styles)
    return pretty_table

def create_pie(df, values, names):
    """
    INPUT:
        df: pandas DataFrame - dataframe for the plot
        x: str - name of column for x axis
        y: str - name of column for y axis

    OUTPUT:
        fig: plotly figure

    Description:
        The function creates customized pie plot
    """
    fig = px.pie(df, values=values, names=names, hole = .4)
    fig.update_layout(template = 'seaborn', height = 600)
    fig.update_traces(textinfo = 'percent+label', rotation = 90,
                      marker = dict(line = dict(color = 'black', width = 2)),
                      insidetextorientation='tangential', opacity = 0.7,
                      outsidetextfont = {'color' : 'black', 'size' : 16, 'family':'Courier New'}, textposition = 'outside',
                      hovertemplate = "%{label}: %{percent} </br>")
    fig.update_layout(
                        template = 'simple_white',
                        title=dict(text='<b>Proportion of sentiments', font_family='Courier New',font_size=16),
                        title_x=0.5,
                        plot_bgcolor='#F0F1D4',
                        paper_bgcolor= '#F3EEF0',
        showlegend=False
                        )
    fig.show()
    return fig

def create_bar(df, x, y):
    """
    INPUT:
        df: pandas DataFrame - dataframe for the plot
        x: str - name of column for x axis
        y: str - name of column for y axis, text

    OUTPUT:
        fig: plotly figure

    Description:
        The function creates customized bar plot
    """
    fig = px.bar(df, x=x, y=y, height=400, text=y)

    fig.update_traces(
                    marker_line_width=2.,
                    marker_color='#38D7A0',
                    marker_line_color='black',
                    texttemplate='%{text:.0f}', textposition='inside',
                    insidetextfont = {'color' : 'black', 'size' : 16},
                    hovertemplate =
                    '<i>Number of listings</i>: %{y:.0f}'
                        )
    fig.update_xaxes(title = dict(text="<b>City</b>", font_family='Courier New',font_size=20),
                     tickfont = dict(size=15, family='Courier New',)
                    )
    fig.update_yaxes(title = dict(text="<b>Number of listings</b>", font_family='Courier New',font_size=20),
                     showgrid=True,
                     tickfont = dict(size=15, family='Courier New',),
                    )
    fig.update_layout(
                        template = 'simple_white',
                        title=dict(text="<b>Number of listings by city</b>", font_family='Courier New',font_size=25),

                        plot_bgcolor='#F0F1D4',
                        paper_bgcolor= '#F3EEF0'
                        )

    fig.show()
    return fig



def create_subplot_bar2(x1, y1, x2, y2):
    """
    INPUT:
        x1: str - name of column for x axis first subplot
        y1: str - name of column for y axis, text first subplot
        x: str - name of column for x axis second subplot
        y: str - name of column for y axis, text second subplot

    OUTPUT:
        fig: plotly figure

    Description:
        The function creates customized bar plot
    """
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Number of Listings by city', 'Population by city'), shared_xaxes=True)
    fig.add_trace(
        go.Bar(x=x1, y=y1, text = y1),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=x2, y=y2, text = y2),
        row=2, col=1
    )
    fig.update_traces(hovertemplate = '<i>Number of listings</i>: %{y:.2s}'+ '<extra></extra>', row=1, col=1)
    fig.update_traces(hovertemplate = '<i>Population</i>: %{y:.2s}'+ '<extra></extra>', row=2, col=1)

    fig.update_traces(
                    marker_line_width=2.,
                    marker_color='#38D7A0',
                    marker_line_color='Black',
                    texttemplate='%{text:.2s}', textposition='inside',
                    insidetextfont = {'color' : 'black', 'size' : 12},
                        )
    fig.update_xaxes(title = dict(text="<b>City</b>", font_family='Courier New',font_size=15),
                        tickfont = dict(size=10, family='Courier New'),row=2,col=1
                    )
    fig.update_yaxes(
        title = dict(text="<b>Number of Listings</b>", font_family='Courier New',font_size=14),
                     showgrid=True,
                     tickfont = dict(size=10, family='Courier New',),row=1,col=1
                    )
    fig.update_yaxes(
        title = dict(text="<b>Population</b>", font_family='Courier New',font_size=14),
                     showgrid=True,
                     tickfont = dict(size=10, family='Courier New',),row=2,col=1
                    )
    fig.update_layout(
                        template = 'simple_white',
                        title=dict(text='', font_family='Courier New',font_size=25),
                        height = 800,
                        plot_bgcolor='#F0F1D4',
                        paper_bgcolor= '#F3EEF0',
                        showlegend=False
                        )
    fig.for_each_annotation(lambda a: a.update(text=f'<b>{a.text}</b>'))
    fig.update_annotations(font=dict(family='Courier New', size=18))
    fig.show()
    return fig
