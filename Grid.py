import os
import time
import json
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
import extra_streamlit_components as stx
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
#from memory_profiler import profile

def get_available_file_paths(directory):
    files = os.listdir(directory)
    file_paths = [os.path.join(directory, file) for file in files]
    return file_paths

def create_line_range_chart(value, title):
    ticks = list(range(1, 11)) 
    tick_descriptions = ['Low Likelihood'] + [''] * 8 + ['High Likelihood']
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[1, 10],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[value],
            y=[0],
            mode='markers',
            marker=dict(size=10, color='red'),
            showlegend=False
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(
            tickvals=ticks,
            ticktext=tick_descriptions,
            tickmode='array',
            showline=True,
            linecolor='black'
        ),
        yaxis=dict(
            showticklabels=False,
            showline=False,
            zeroline=False
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

def create_donut(hex_colors, css_colors, percentages, title):
    hover_info = [f'Color: {css}<br>Hex: {hex_color}<br>Percentage: {percentage:.1f}%' 
              for css, hex_color, percentage in zip(css_colors, hex_colors, percentages)]
    
    fig = go.Figure(data=[go.Pie(
        labels=hover_info,
        values=percentages,
        #textinfo='none',  
        hoverinfo='text+percent+label',
        hole=0.5,
        marker=dict(colors=hex_colors),
        showlegend=False,
    )])
    
    fig.update_layout(
        title={
            'text': title,
            'font_size': 24, 
            'x': 0.25, 
            'y': 0.95
        },
        width=500,
        height=500,
        #annotations=[dict(text='Donut Chart', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    return fig

#@profile
def app():
    # ----------------------
    # UI SETUP
    # ----------------------
    
    st.set_page_config(layout="wide")
    sidebar = st.sidebar
    st.session_state['search_data'] = {}

    st.sidebar.markdown(
        '<h1 style="color: #92B9FF;; text-align: left;">Configurations</h1>',
        unsafe_allow_html=True
    )

    custom_title = """
    <div style="font-size: 36px; color: #92B9FF; font-family: 'Verdana', sans-serif; margin-top: -50px;">
        Content Pipeline - Interact with your Data
    </div>
    """
    st.markdown(custom_title, unsafe_allow_html=True)
    
    # set directory path relative to current directory
    directory_path = os.path.join(os.getcwd(), 'images')
    json_directory_path = os.path.join(os.getcwd(), 'json_files')
    options = get_available_file_paths(directory_path)
    options = sorted(options, key=lambda x: x.split('/')[-1])

    if 'images' not in st.session_state:
        images = get_available_file_paths(directory_path)
        filenames = [image.split('/')[-1] for image in images]
        images = [Image.open(image) for image in images]
        st.session_state['images'] = {
            'images': images,
            'filenames': filenames
        }

    reset_button = sidebar.button("Reset")
   
    if reset_button:
        st.rerun()

    st.write("")
    st.write("")

    #num_cols = 4
    #columns = st.columns(spec=num_cols, gap='large')
    #for i, image in enumerate(st.session_state['images']['images']):
    #    with columns[i % num_cols]:
    #        with st.container():
    #            image = image.resize((200, 300), Image.ANTIALIAS)
    #            st.image(image, caption=st.session_state['images']['filenames'][i], use_column_width=True)

if __name__ == "__main__":
    app()