import streamlit as st
from streamlit_extras.let_it_rain import rain
# 爱心代码 ，伴有动画效果，爱心下落
st.balloons()
rain(emoji="💕", font_size=54,
     falling_speed=5, animation_length="infinite", )
