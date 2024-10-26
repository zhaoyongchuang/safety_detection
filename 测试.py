import streamlit as st

# // 设定3列
col1, col2, col3 = st.columns(3)

# // 设定不同的列标题和展示的内容
with col1: # // 第一列
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg")