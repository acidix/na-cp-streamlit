import argparse
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

    parser = argparse.ArgumentParser(description='content_pipeline', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_directory', default=None, type=str)
    parser.add_argument('--json_directory', default=None, type=str)
    args = parser.parse_args()

    # ----------------------
    # UI SETUP
    # ----------------------
    st.set_page_config(layout="wide")
    
    custom_title = """
    <div style="font-size: 36px; color: #92B9FF; font-family: 'Verdana', sans-serif; margin-top: -50px;">
        Content Pipeline - Interact with your Data
    </div>
    """
    st.markdown(custom_title, unsafe_allow_html=True)
    
    directory_path = args.image_directory
    json_directory_path = args.json_directory
    options = get_available_file_paths(directory_path)
    options = sorted(options, key=lambda x: x.split('/')[-1])

    sidebar = st.sidebar
    st.sidebar.markdown(
        '<h1 style="color: #92B9FF;; text-align: left;">Configurations</h1>',
        unsafe_allow_html=True
    )
    chosen_id = stx.tab_bar(data=[
        stx.TabBarItemData(id=1, title="Image Grid View", description=""),
        stx.TabBarItemData(id=2, title="Image Analysis", description=""),
        stx.TabBarItemData(id=3, title="Use Case: Image Search", description=""),
    ], default=1)

    args.image_name = sidebar.selectbox('Select an image', [option.split('/')[-1] for option in options])
    data_category = ['OCR', 'Image Parts', 'Labels', 'Face Analysis', 'Safe Search', 'Image Properties', 'Logo Detection', 'Landmark Detection', 'Web Detection']
    data_to_visualize = sidebar.selectbox('Select content category to visualize', data_category)

    annotate_button = sidebar.button("Annotate Image")
    reset_button = sidebar.button("Reset")
    
    if 'images' not in st.session_state:
        images = get_available_file_paths(directory_path)
        filenames = [image.split('/')[-1] for image in images]
        images = [Image.open(image) for image in images]
        st.session_state['images'] = {
            'images': images,
            'filenames': filenames
        }
    
    if 'search_data' not in st.session_state:
        st.session_state['search_data'] = {}

    if reset_button:
        st.session_state['search_data'] = {}
        st.experimental_rerun()

    if chosen_id == '1':
        st.write("")
        st.write("")

        num_cols = 4
        columns = st.columns(spec=num_cols, gap='medium')
        for i, image in enumerate(st.session_state['images']['images']):
            with columns[i % num_cols]:
                with st.container():
                    image = image.resize((200, 300))
                    st.image(image, caption=st.session_state['images']['filenames'][i], use_column_width=False)
                
    if chosen_id == '2':
        image = Image.open(os.path.join(directory_path, args.image_name))
        columns = st.columns(spec=2, gap='medium')
        columns[0] = columns[0].empty()

        with columns[0].container():
            st.image(image, caption='Uploaded Image', use_column_width=True)

        with columns[1]:
            if annotate_button:
                with st.spinner(f"Annotating the image..."):
                    json_file = args.image_name.split('/')[-1].split('.')[0] + '.json'
                    json_path = os.path.join(json_directory_path, json_file)
                    with open(json_path) as f:
                        data = json.load(f)

                    if data_to_visualize == 'OCR':                                
                        recognized_words = data['full_image_content']['OCR']
                        words = []
                        for recognized_word in recognized_words.values():
                            word = recognized_word['text']
                            width = recognized_word['geometry']['BoundingBox']['Width']
                            height = recognized_word['geometry']['BoundingBox']['Height']
                            left = recognized_word['geometry']['BoundingBox']['Left']
                            top = recognized_word['geometry']['BoundingBox']['Top']
                            image_width, image_height = image.size
                            x1 = int(left * image_width)
                            y1 = int(top * image_height)
                            x2 = int((left + width) * image_width)
                            y2 = int((top + height) * image_height)
                            draw = ImageDraw.Draw(image)
                            draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=3)
                            words.append(word)

                        time.sleep(0.3)
                        with columns[0].container():
                            st.image(image, caption='Annotated Image', use_column_width=True)
                        st.write(f'OCR Output: {len(words)} words found')
                        words = ' '.join(words)
                        words = words.split()
                        words = [word if (i+1) % 8 != 0 else word + '\n' for i, word in enumerate(words)]
                        words = ' '.join(words)
                        st.code(words, language='text')
                
                    elif data_to_visualize == 'Image Parts':
                        recognized_parts = [key for key in data.keys() if key.startswith(args.image_name.split('/')[-1].split('.')[0] + '_recognized_part')]
                        for recognized_part in recognized_parts:
                            part = data[recognized_part]
                            box = part['bounding_box']
                            width = box[0]
                            height = box[1]
                            left = box[2]
                            top = box[3]
                            
                            draw = ImageDraw.Draw(image)
                            draw.rectangle([(width, height), (left, top)], outline='green', width=20)
                        time.sleep(0.3)
                        with columns[0].container():
                            st.image(image, caption='Annotated Image', use_column_width=True)
                        image.close()
                        st.write(f'Image Parts Output: {len(recognized_parts)} parts found')
                        for recognized_part in recognized_parts:
                            image = Image.open(os.path.join(directory_path, args.image_name))
                            part = data[recognized_part]
                            box = part['bounding_box']
                            width = box[0]
                            height = box[1]
                            left = box[2]
                            top = box[3]
                            image_part = image.crop((width, height, left, top))
                            st.image(image_part, caption=recognized_part, use_column_width=True)
                            image.close()
                    
                    elif data_to_visualize == 'Labels':
                        if not "LABEL_DETECTION" in data['full_image_content']:
                            pass
                        else:
                            labels = data['full_image_content']['LABEL_DETECTION']
                            dataframe = pd.DataFrame(columns=['Image', 'Label', 'Score'])
                            for label in labels:
                                dataframe = pd.concat([dataframe, pd.DataFrame([{'Image': image, 'Label': label['label_name'], 'Score': label['label_confidence'], 'bounding_boxes': label['label_boundings']}])], ignore_index=True)
                            dataframe = dataframe.sort_values(by=['Score'], ascending=False)
                            dataframe = dataframe.reset_index(drop=True)
                            dataframe['Score'] = dataframe['Score'].round(2)
                            with st.container():
                                display_cols = st.columns(spec=3, gap='medium')
                                with display_cols[0]:
                                    st.image(image, use_column_width=True)
                                    dataframe_no_bb = [row for index, row in dataframe.iterrows() if len(row['bounding_boxes']) == 0]
                                    labels = [row['Label'] for row in dataframe_no_bb]
                                    labels = [label if (i+1) % 2 != 0 else label + '\n' for i, label in enumerate(labels)]
                                    scores = [row['Score'] for row in dataframe_no_bb]
                                    scores = [str(score) if (i+1) % 2 != 0 else str(score) + '\n' for i, score in enumerate(scores)]
                                    labels = ' / '.join(labels)
                                    scores = ' / '.join([str(score) for score in scores])
                                    display_cols[1].code(labels)
                                    display_cols[2].code(scores)
                                st.divider()

                            dataframe_only_bb = [row for index, row in dataframe.iterrows() if len(row['bounding_boxes']) != 0]
                            for index, row in enumerate(dataframe_only_bb):

                                label = row['Label']
                                score = row['Score']
                                bounding_boxes = row['bounding_boxes']
                                for bounding_box in bounding_boxes:
                                    container = st.container()
                                    display_cols = container.columns(spec=3, gap='medium')
                                    image = Image.open(os.path.join(directory_path, args.image_name))
                                    width = bounding_box['Width']
                                    height = bounding_box['Height']
                                    left = bounding_box['Left']
                                    top = bounding_box['Top']
                                    image_width, image_height = image.size
                                    x1 = int(left * image_width)
                                    y1 = int(top * image_height)
                                    x2 = int((left + width) * image_width)
                                    y2 = int((top + height) * image_height)
                                    image_part = image.crop((x1, y1, x2, y2))
                                    with display_cols[0]:
                                        st.image(image_part, use_column_width=True)
                                    with display_cols[1]:
                                        st.code(label)
                                    with display_cols[2]:
                                        st.code(str(score))
                                    image.close()     
                                    container.divider()  

                            recognized_parts = [key for key in data.keys() if key.startswith(args.image_name.split('/')[-1].split('.')[0] + '_recognized_part')]
                            for recognized_part in recognized_parts:
                                container = st.container()
                                display_cols = container.columns(spec=3, gap='medium')
                                part = data[recognized_part]
                                if not 'LABEL_DETECTION' in part.keys():
                                    continue
                                box = part['bounding_box']
                                width = box[0]
                                height = box[1]
                                left = box[2]
                                top = box[3]
                                
                                image = Image.open(os.path.join(directory_path, args.image_name))
                                image_part = image.crop((width, height, left, top))
                                
                                with display_cols[0]:
                                    st.image(image_part, use_column_width=True)
                                image.close()

                                with display_cols[1]:
                                    labels = [label['label_name'] for label in part['LABEL_DETECTION']]
                                    labels = [label if (i+1) % 2 != 0 else label + '\n' for i, label in enumerate(labels)]
                                    labels = ' / '.join(labels)
                                    st.code(labels)
                                
                                with display_cols[2]:
                                    scores = [label['label_confidence'] for label in part['LABEL_DETECTION']]
                                    scores = [str(round(score,2)) if (i+1) % 2 != 0 else str(round(score,2)) + '\n' for i, score in enumerate(scores)]
                                    scores = ' / '.join(scores)
                                    st.code(scores)
                                container.divider()

                    elif data_to_visualize == 'Face Analysis':
                        if not "FACE_DETECTION" in data['full_image_content']:
                            pass
                        else:
                            data = data['full_image_content']['FACE_DETECTION']
                            for elem in data:
                                container = st.container()
                                display_cols = container.columns(spec=3, gap='medium')
                                bounding_box = elem['face_boundings']
                                width = bounding_box['Width']
                                height = bounding_box['Height']
                                left = bounding_box['Left']
                                top = bounding_box['Top']

                                image_width, image_height = image.size
                                x1 = int(left * image_width)
                                y1 = int(top * image_height)
                                x2 = int((left + width) * image_width)
                                y2 = int((top + height) * image_height)
                                image = Image.open(os.path.join(directory_path, args.image_name))
                                image_part = image.crop((x1, y1, x2, y2))
                                with display_cols[0]:
                                    st.image(image_part, use_column_width=True)
                                image.close()

                                smile = elem['face_has_smile']
                                smile_confidence = elem['face_smile_confidence']
                                eyeglasses = elem['face_has_eyeglasses']
                                eyeglasses_confidence = elem['face_eyeglasses_confidence']
                                sunglasses = elem['face_has_sunglasses']
                                sunglasses_confidence = elem['face_sunglasses_confidence']
                                beard = elem['face_has_beard']
                                beard_confidence = elem['face_beard_confidence']
                                mustache = elem['face_has_mustache']
                                mustache_confidence = elem['face_mustache_confidence']
                                eyes_open = elem['face_has_eyes_open']
                                eyes_open_confidence = elem['face_eyes_open_confidence']
                                mouth_open = elem['face_has_mouth_open']
                                mouth_open_confidence = elem['face_mouth_open_confidence']

                                attributes = [f'Smiling: {smile}', f'Eyeglasses: {eyeglasses}', f'Sunglasses: {sunglasses}', f'Beard: {beard}', f'Mustache: {mustache}', f'Eyes Open: {eyes_open}', f'Mouth Open: {mouth_open}']
                                attributes = [attribute if (i+1) % 2 != 0 else attribute + '\n' for i, attribute in enumerate(attributes)]
                                attributes = ' / '.join(attributes)

                                confidences = [f'Smiling: {round(smile_confidence,2)}', f'Eyeglasses: {round(eyeglasses_confidence,2)}', f'Sunglasses: {round(sunglasses_confidence,2)}', f'Beard: {round(beard_confidence,2)}', f'Mustache: {round(mustache_confidence,2)}', f'Eyes Open: {round(eyes_open_confidence,2)}', f'Mouth Open: {round(mouth_open_confidence,2)}']
                                confidences = [confidence if (i+1) % 2 != 0 else confidence + '\n' for i, confidence in enumerate(confidences)]
                                confidences = ' / '.join(confidences)   
                                with display_cols[1]:
                                    st.code(attributes)
                                with display_cols[2]:
                                    st.code(confidences)
                                container.divider()
                    
                    elif data_to_visualize == 'Web Detection':
                        if not "WEB_DETECTION" in data['full_image_content']:
                            pass
                        else:
                            web_entities = data['full_image_content']['WEB_DETECTION']['web_entities']
                            columns[1].write(f'Web Entities Output: {len(web_entities)} entities found')
                            for elem in web_entities:
                                with st.container():
                                    columns[1].code(elem['web_entity_description'])
                                    columns[1].divider()

                            web_visually_similar_images = data['full_image_content']['WEB_DETECTION']['web_visually_similar_images']
                            columns[1].write(f'Web Visually Similar Images Output: {len(web_visually_similar_images)} images found')
                            for elem in web_visually_similar_images:
                                with st.container():
                                    columns[1].image(elem['web_visually_similar_image_url'], caption=elem['web_visually_similar_image_url'], use_column_width=True)    
                                    columns[1].divider()
                            
                            web_best_guess_labels = data['full_image_content']['WEB_DETECTION']['web_best_guess_labels']
                            columns[1].write(f'Web Best Guess Labels Output: {len(web_best_guess_labels)} labels found')
                            for elem in web_best_guess_labels:
                                with st.container():
                                    columns[1].code(elem['web_best_guess_label'])
                                    columns[1].divider()
                    
                    elif data_to_visualize == 'Image Properties':
                        if not "IMAGE_PROPERTIES" in data['full_image_content'].keys():
                            pass
                        else:
                            image_properties = data['full_image_content']['IMAGE_PROPERTIES']
                            categories = ['dominant_color', 'foreground_dominant_color', 'background_dominant_color']
                            plot_labels = ['Overall Dominant Colors', 'Foreground Dominant Colors', 'Background Dominant Colors']
                            for i, category in enumerate(categories):
                                dominant_colors_hex = []
                                dominant_colors_css = []
                                dominant_colors_percent = []
                                for j, elem in enumerate(image_properties[0].keys()):
                                    if f'{category}_{j}_hex' in image_properties[0].keys():
                                        dominant_colors_hex.append(image_properties[0][f'{category}_{j}_hex'])
                                        dominant_colors_css.append(image_properties[0][f'{category}_{j}_css'])
                                        dominant_colors_percent.append(image_properties[0][f'{category}_{j}_pixelpercent'])
                                
                                fig = create_donut(dominant_colors_hex, dominant_colors_css, dominant_colors_percent, plot_labels[i])
                                st.plotly_chart(fig)
                                st.divider()
                    
                    elif data_to_visualize == 'Safe Search':
                        if not "SAFE_SEARCH_DETECTION" in data['full_image_content'].keys():
                            pass
                        else:
                            safe_search = data['full_image_content']['SAFE_SEARCH_DETECTION']
                            adult = safe_search['adult']
                            medical = safe_search['medical']
                            spoof = safe_search['spoof']
                            violence = safe_search['violence']
                            racy = safe_search['racy']
                            fig = create_line_range_chart(adult, "Contains Adult Content?")
                            st.plotly_chart(fig)

    if chosen_id == '3':
        search_query = st.text_input('Enter a search query by providing relevant terms for your image', 'cat')
        search_button = st.button('Search related images')

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

        if search_button:
            images = get_available_file_paths(directory_path)
            filenames = [image.split('/')[-1] for image in images]
            images = [Image.open(image) for image in images]
            vectorizer = CountVectorizer()
            corpus = []
            for key in st.session_state['search_data'].keys():
                corpus.append(' '.join(st.session_state['search_data'][key]))
            X = vectorizer.fit_transform(corpus)
            X = X.toarray()
            search_query_vectorized = vectorizer.transform([search_query])
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
            




                            



                                    



                    