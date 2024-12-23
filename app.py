import streamlit as st
import re
import base64
from io import BytesIO
import requests
from PIL import Image
from langchain_ollama import ChatOllama
import ollama

st.set_page_config(layout="wide")

st.title("Ollama Test")

def clear() -> None:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
    ]

def rewind() -> None:
    if len(st.session_state.messages) >= 1:
        st.session_state.messages = st.session_state.messages[:-1]

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def convert_to_pil(img_str):
    img_data = re.sub('^data:image/.+;base64,', '', img_str)
    img_pil = Image.open(BytesIO(base64.b64decode(img_data)))
    return img_pil

if "messages" not in st.session_state:
    clear()

@st.cache_resource
def _load_llm(model_name:str, temperature:float, top_k:int, top_p:float):
    llm = ChatOllama(model=model_name, temperature=temperature, top_k=top_k, top_p=top_p)
    return llm

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox("model", [m.model for m in ollama.list().models])
    _is_vision = (model_name.split(':')[0].split('-')[-1] == "vision")
    temperature = st.slider("temperature", min_value=0.0, max_value=2.0, value=0.8)
    top_k = st.number_input("top_k", min_value=1, value=40)
    top_p = st.slider("top_p", min_value=0.0, max_value=1.0, value=0.9)

    st.header("Chat Control")
    role = st.selectbox("role", ['human', 'assistant', 'system'])

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        else:
            for content in message["content"]:
                if content["type"] == "image_url":
                    img = convert_to_pil(content["image_url"])
                    st.image(img)
                else:
                    st.markdown(content["text"])

chat_area = st.container()
def gen_res():
    with chat_area:
        with st.chat_message("assistant"):
            llm = _load_llm(model_name, temperature, top_k, top_p)
            response = st.write_stream(llm.stream(st.session_state.messages))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.button("Rewind", on_click=rewind, use_container_width=True, type='primary')
    with btn_col2:
        st.button("Clear", on_click=clear, use_container_width=True)
    st.button("Generate", on_click=gen_res, use_container_width=True)

    if (role == 'human') and _is_vision:
        st.header("Vision")
        image_base64 = None
        url = st.text_input("Image URL", key=f"url_{len(st.session_state.messages)}")
        if url:
            response = requests.get(url)
            _image = Image.open(BytesIO(response.content))
            st.image(_image)
            image_base64 = f"data:image/png;base64,{convert_to_base64(_image)}"
        else:
            image = st.file_uploader("Upload an image", type=["jpg","png","jpeg"], key=f"img_{len(st.session_state.messages)}")
            if image:
                with st.chat_message(role):
                    st.image(image)
                image_base64 = f"data:image/png;base64,{convert_to_base64(Image.open(image))}"

if prompt := st.chat_input("Your message"):
    with chat_area:
        # Display input message in chat message container
        with st.chat_message(role):
            st.markdown(prompt)
        # Add input message to chat history
        if (role == 'human') and _is_vision and image_base64:
            st.session_state.messages.append({"role": role, "content": [
                {
                    "type": "image_url",
                    "image_url": image_base64,
                },
                {"type": "text", "text": prompt},
            ]})
        else:
            st.session_state.messages.append({"role": role, "content": prompt})

    # LLM response
    if role == "human":
        gen_res()
    st.rerun()