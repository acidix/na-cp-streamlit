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

if __name__ == "__main__":

    # ----------------------
    # UI SETUP
    # ----------------------
    
    st.set_page_config(layout="wide")
    sidebar = st.sidebar
    st.session_state['search_data'] = {}

    st.sidebar.markdown(
        '<h1 style="color: #92B9FF;; text-align: left;">Search</h1>',
        unsafe_allow_html=True
    )

    custom_title = """
    <div style="font-size: 36px; color: #92B9FF; font-family: 'Verdana', sans-serif; margin-top: -50px;">
        Search for Images
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

    search_term = sidebar.text_input('Enter a search term', 'man with glasses')

    search_button = sidebar.button("Search")

    st.session_state['show_search'] = True
    st.session_state['show_grid'] = False
    st.session_state['show_annotation'] = False

    st.markdown(
        f'''<h1 style="color: #92B9FF;; text-align: left;">Search Results</h1>
        <h3 style="color: #92B9FF;; text-align: left;">Top 5 Images for search term: {search_term}</h3>''',
        unsafe_allow_html=True
    )

    if st.session_state['search_data'] == {}:
        images = get_available_file_paths(directory_path)
        filenames = [image.split('/')[-1] for image in images]
        images = [Image.open(image) for image in images]
        for i, image in enumerate(images):
            st.session_state['search_data'][filenames[i]] = []
            json_file = image.filename.split('/')[-1].split('.')[0] + '.json'
            json_path = os.path.join(json_directory_path, json_file)
            with open(json_path) as f:
                data = json.load(f)
            if not "LABEL_DETECTION" in data['full_image_content']:
                pass
            else:
                labels = data['full_image_content']['LABEL_DETECTION']
                for label in labels:
                    st.session_state['search_data'][filenames[i]].append(label['label_name'])

            if not "FACE_DETECTION" in data['full_image_content']:
                pass
            else:
                faces = data['full_image_content']['FACE_DETECTION']
                for face in faces:
                    st.session_state['search_data'][filenames[i]].append('Smiling' if face['face_has_smile'] else 'Not Smiling')
                    st.session_state['search_data'][filenames[i]].append('Eyes Open' if face['face_has_eyes_open'] else 'Eyes Closed')
                    st.session_state['search_data'][filenames[i]].append('Mouth Open' if face['face_has_mouth_open'] else 'Mouth Closed')
                    st.session_state['search_data'][filenames[i]].append('Eyeglasses' if face['face_has_eyeglasses'] else 'No Eyeglasses')
                    st.session_state['search_data'][filenames[i]].append('Sunglasses' if face['face_has_sunglasses'] else 'No Sunglasses')
                    st.session_state['search_data'][filenames[i]].append('Beard' if face['face_has_beard'] else 'No Beard')
                    st.session_state['search_data'][filenames[i]].append('Mustache' if face['face_has_mustache'] else 'No Mustache')
                    st.session_state['search_data'][filenames[i]].append('Male' if face['face_gender'] == 'Male' else 'Female')
        for key in st.session_state['search_data'].keys():
            st.session_state['search_data'][key] = list(set(st.session_state['search_data'][key]))

    images = get_available_file_paths(directory_path)
    filenames = [image.split('/')[-1] for image in images]
    images = [Image.open(image) for image in images]
    vectorizer = CountVectorizer()
    corpus = []
    for key in st.session_state['search_data'].keys():
        corpus.append(' '.join(st.session_state['search_data'][key]))
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    search_query_vectorized = vectorizer.transform([search_term])
    search_query_vectorized = search_query_vectorized.toarray()
    cosine_similarities = cosine_similarity(search_query_vectorized, X)
    cosine_similarities = cosine_similarities[0]
    indices = list(range(len(cosine_similarities)))
    indices = sorted(indices, key=lambda x: cosine_similarities[x], reverse=True)
    cosine_similarities = sorted(cosine_similarities, reverse=True)
    top_5_images = []
    for i in range(5):
        top_5_images.append(images[indices[i]])
    top_5_images = [image.resize((200, 300)) for image in top_5_images]
    columns = st.columns(spec=5, gap='medium')

    for i, image in enumerate(top_5_images):
        with columns[i]:
            with st.container():
                st.image(image, caption=filenames[indices[i]], use_column_width=False)
                st.write(f'Cosine Similarity: {cosine_similarities[i]}')
                #st.write(f'Labels: {st.session_state["search_data"][filenames[indices[i]]]}')


                        



                                



                
