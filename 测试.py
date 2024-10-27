# coding:utf-8
import streamlit as st
from markdown_it.rules_inline import backtick
from streamlit_extras.let_it_rain import rain
# 爱心代码 ，伴有动画效果，爱心下落
# bk_css ="""
#     <style>
#     body {
#         background-image: url("background.jpg");
#         # background-size: cover;
#     }
#     </style>
#     """
# 全屏显示background.jpg
st.title("I love diandian")
# st.markdown(bk_css,unsafe_allow_html=True)
st.image("background.jpg")
# st.markdown(backtick_)
st.balloons()
rain(emoji="💕", font_size=54,
     falling_speed=5, animation_length="infinite", )
# 全屏显示background.jpg
