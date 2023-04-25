from track import *
import tempfile
import cv2
import torch
import streamlit as st
import os


if __name__ == '__main__':
    st.title('Détection et dénombrement de véhicules ciculant sur un pont.')
    st.markdown('<h3 style="color: red"> avec Yolov5 et Deep SORT </h3', unsafe_allow_html=True)

    # upload video
    video_file_buffer = st.sidebar.file_uploader("Garroche moi ta video", type=['mp4', 'mov', 'avi'])

    if video_file_buffer:
        st.sidebar.text('Video')
        st.sidebar.video(video_file_buffer)
        # save video from streamlit into "videos" folder for future detect
        with open(os.path.join('videos', video_file_buffer.name), 'wb') as f:
            f.write(video_file_buffer.getbuffer())

    st.sidebar.markdown('---')
    st.sidebar.title('Patentes')
    # custom class
    custom_class = st.sidebar.checkbox('Classes d\'objets')
    assigned_class_id = [0, 1, 2, 3]
    names = ['car', 'motorcycle', 'truck', 'bus']

    if custom_class:
        assigned_class_id = []
        assigned_class = st.sidebar.multiselect('Choisis des classes', list(names))
        for each in assigned_class:
            assigned_class_id.append(names.index(each))
    
    # st.write(assigned_class_id)

    # setting hyperparameter
    confidence = st.sidebar.slider('Confiance', min_value=0.0, max_value=1.0, value=0.5)
    line = st.sidebar.number_input('Position de la ligne', min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    st.sidebar.markdown('---')

    
    status = st.empty()
    stframe = st.empty()
    if video_file_buffer is None:
        status.markdown('<font size= "4"> **Status:** En attente du vidéo </font>', unsafe_allow_html=True)
    else:
        status.markdown('<font size= "4"> **Status:** Pret </font>', unsafe_allow_html=True)

    car, bus, truck, motor = st.columns(4)
    with car:
        st.markdown('**Autos**')
        car_text = st.markdown('__')
    
    with bus:
        st.markdown('**Bus**')
        bus_text = st.markdown('__')

    with truck:
        st.markdown('**Camion**')
        truck_text = st.markdown('__')
    
    with motor:
        st.markdown('**Moto**')
        motor_text = st.markdown('__')

    fps, _,  _, _  = st.columns(4)
    with fps:
        st.markdown('**FPS**')
        fps_text = st.markdown('__')


    track_button = st.sidebar.button('AWAILLE!')
    # reset_button = st.sidebar.button('RESET ID')
    if track_button:
        # reset ID and count from 0
        reset()
        opt = parse_opt()
        opt.conf_thres = confidence
        opt.source = f'videos/{video_file_buffer.name}'

        status.markdown('<font size= "4"> **Status:** ça roule ma poule... </font>', unsafe_allow_html=True)
        with torch.no_grad():
            detect(opt, stframe, car_text, bus_text, truck_text, motor_text, line, fps_text, assigned_class_id)
        status.markdown('<font size= "4"> **Status:** Fini ! </font>', unsafe_allow_html=True)
        # end_noti = st.markdown('<center style="color: blue"> FINISH </center>',  unsafe_allow_html=True)

    # if reset_button:
        # reset()
    #     st.markdown('<h3 style="color: blue"> Reseted ID </h3>', unsafe_allow_html=True)
