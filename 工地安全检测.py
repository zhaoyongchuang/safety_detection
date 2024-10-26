# coding:utf-8
# Ultralytics YOLO 🚀, AGPL-3.0 license
import io
import time
import os
import cv2
import torch
import smtplib
from ultralytics.utils.checks import check_requirements
from streamlit_extras.let_it_rain import rain
# from streamlit_extras.app_logo import add_logo

from_email = "m"  # must match the email used to generate the password
to_email = ""
password = ""  # qq邮箱授权码
server = smtplib.SMTP("imap.qq.com", 587)  #
server.starttls()
server.login(from_email, password)
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(to_email, from_email, object_detected=1):
    """Sends an email notification indicating the number of objects detected; defaults to 1 object."""
    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = "Security Alert"
    # Add in the message body
    message_body = f"警告！！！有物体进入禁区 \n{object_detected}个物体被检测到！！！"

    message.attach(MIMEText(message_body, "plain"))
    server.sendmail(from_email, to_email, message.as_string())


def inference(model=None):
    """Runs real-time object detection on video input using Ultralytics YOLOv8 in a Streamlit application."""
    email_sent = False  # Add this line
    inter_start_time = time.time() - 31  # 间隔30秒发送邮件
    check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds
    import streamlit as st

    from ultralytics import YOLO

    # Hide main menu style
    menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # 隐藏主菜单样式

    # Main title of streamlit application
    main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:35px; 
                             font-family: 'Archivo', sans-serif; margin-top:-50px;margin-bottom:20px;">
                    基于YOLOv8+Streamlit的工地安全检测平台
                    </h1></div>"""

    # Subtitle of streamlit application
    sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center;
                    font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;">
                    Experience real-time object detection on your webcam with the power of Ultralytics YOLOv8! 🚀</h4>
                    </div>"""

    # Set html page configuration
    st.set_page_config(page_title="安徽大学 工地安全检测平台", layout="wide", initial_sidebar_state="auto",
                       page_icon="./pic/ahu.svg")

    # Append the custom HTML
    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    st.markdown(main_title_cfg, unsafe_allow_html=True)
    # st.markdown(sub_title_cfg, unsafe_allow_html=True)

    # Add ultralytics logo in sidebar
    with st.sidebar:
        st.title("安徽大学 工地安全检测平台")
        # logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
        logo = "D:/deeplearning_stu/streamlit_stu/pic/ahu.svg"
        # 居中显示
        # st.image(logo, width=70)
        st.logo(logo)

    # Add elements to vertical setting menu
    st.sidebar.title("用户自定义配置", anchor="top")

    # Add video source selection dropdown
    source = st.sidebar.selectbox(
        "Video",
        ("webcam", "video"),
    )

    vid_file_name = ""
    if source == "video":
        vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
        if vid_file is not None:
            g = io.BytesIO(vid_file.read())  # BytesIO Object
            vid_location = "ultralytics.mp4"
            with open(vid_location, "wb") as out:  # Open temporary file as bytes
                out.write(g.read())  # Read bytes into file
            vid_file_name = "ultralytics.mp4"
    elif source == "webcam":
        vid_file_name = 0

        # cap = cv2.VideoCapture(url)


    # Add dropdown menu for model selection
    # available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolov8")]
    # if model:
    #     available_models.insert(0, model.split(".pt")[0])  # insert model without suffix as *.pt is added later

    # selected_model = st.sidebar.selectbox("Model", available_models)
    # with st.spinner("Model is downloading..."):
    # model = YOLO(f"{selected_model.lower()}.pt")  # Load the YOLO model
    # 单选框，从./yolo8_weights文件夹中选择模型
    weight_list = os.listdir("D:/deeplearning_stu/yolo8_weight")
    # 设置一个下拉框，选择模型,默认是yolov8n
    selected_model = st.sidebar.selectbox("Model", weight_list, index=2)
    # 加载模型
    model = YOLO(f"D:/deeplearning_stu/yolo8_weight/{selected_model}")
    # 模型支持的类别
    class_names = list(model.names.values())  # Convert dictionary to list of class names
    # st.success("Model loaded successfully!")

    # Multiselect box with class names and get indices of selected classes
    selected_classes = st.sidebar.multiselect("Classes", class_names, default=["person", "cup", "wine glass", "bottle"])
    selected_ind = [class_names.index(option) for option in selected_classes]

    if not isinstance(selected_ind, list):  # Ensure selected_options is a list
        selected_ind = list(selected_ind)

    enable_trk = st.sidebar.radio("是否启用跟踪", ("Yes", "No")) # 是否启用跟踪
    conf = float(st.sidebar.slider("过滤低置信度的边界框", 0.0, 1.0, 0.25, 0.01))  # 这个conf是指置信度，用于过滤低置信度的边界框
    iou = float(st.sidebar.slider("过滤重叠的边界框", 0.0, 1.0, 0.45, 0.01))  # 这个iou是指交并比，用于过滤重叠的边界框

    col1, col2 = st.columns(2)
    org_frame = col1.empty()
    ann_frame = col2.empty()

    fps_display = st.sidebar.empty()  # Placeholder for FPS display

    cap = cv2.VideoCapture(vid_file_name)
    frame_placeholder = st.empty()
    start_work = st.sidebar.button("Start")
    while cap.isOpened() and not start_work:
        ret, frame = cap.read()
        if not ret:
            st.write("Video capture has ended.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        if start_work:
            break


    if start_work:
        videocapture = cv2.VideoCapture(vid_file_name)  # Capture the video

        if not videocapture.isOpened():
            st.error("Could not open webcam.")

        stop_button = st.button("Stop")  # Button to stop the inference

        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                st.warning("Failed to read frame from webcam. Please make sure the webcam is connected properly.")
                break

            prev_time = time.time()

            # Store model predictions
            if enable_trk == "Yes":
                results = model.track(frame, conf=conf, iou=iou, classes=selected_ind, persist=True)
            else:
                results = model(frame, conf=conf, iou=iou, classes=selected_ind)
            annotated_frame = results[0].plot()  # Add annotations on frame
            # 如果检测到物体，弹窗提醒
            if len(results[0].boxes.cls) > 0:
                # st.warning("有物体闯入！！！", icon="⚠️")
                # rain(emoji="⚠️", font_size=54,
                #      falling_speed=5, animation_length="infinite", )
                curr_time = time.time()
                if curr_time - inter_start_time > 30:
                    # send_email(to_email, from_email, len(results[0].boxes.cls))
                    # rain(emoji="⚠️", font_size=54,
                    #      falling_speed=5, animation_length="infinite", )

                    inter_start_time = time.time()

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # display frame
            org_frame.image(frame, channels="BGR")
            ann_frame.image(annotated_frame, channels="BGR")

            if stop_button:
                videocapture.release()  # Release the capture
                torch.cuda.empty_cache()  # Clear CUDA memory
                st.stop()  # Stop streamlit app

            # Display FPS in sidebar
            fps_display.metric("FPS", f"{fps:.2f}")

        # Release the capture
        videocapture.release()

    # Clear CUDA memory
    torch.cuda.empty_cache()

    # Destroy window
    cv2.destroyAllWindows()


# Main function call

if __name__ == "__main__":
    inference()

