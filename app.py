import streamlit as st
import pytesseract
import spacy
import ollama
from PIL import Image
import json
import base64
import re
import string

# Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'd:\tesseract\tesseract.exe'

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# OCR function
def extract_text_from_image(image):
    extracted_text = pytesseract.image_to_string(image)
    lines = extracted_text.splitlines()
    instruction_lines = [
        re.sub(r"[^a-zA-Z,\s]", "", line.strip())
        for line in lines if line.strip()
    ]
    return instruction_lines

# Preprocessing function
def preprocess_sentences(sentences):
    split_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        current_chunk = []
        for token in doc:
            current_chunk.append(token.text)
            if token.dep_ in {"cc", "punct"} and token.text in {";", "or", "but"}:
                if current_chunk:
                    split_sentences.append(" ".join(current_chunk).strip())
                    current_chunk = []
        if current_chunk:
            split_sentences.append(" ".join(current_chunk).strip())
    return [re.sub(r'\s*,\s*', ',', sentence) for sentence in split_sentences]

# Generate Mermaid.js code using POS tagging
def parse_instructions(sentences):
    nodes = ['A["Start"]']
    edges = []
    node_map = {}
    label_sequence = iter(string.ascii_uppercase)
    prev_node_id = next(label_sequence)

    for sentence in sentences:
        doc = nlp(sentence)
        verb_phrase = None
        noun_phrases = []

        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                verb_phrase = token.lemma_.capitalize()
                for child in token.children:
                    if child.dep_ in {"prep", "advmod", "dobj"}:
                        verb_phrase += f" {child.text}"

        for chunk in doc.noun_chunks:
            noun_phrases.append(chunk.text.strip())

        node_label = ", ".join(noun_phrases)
        if node_label not in node_map:
            node_id = next(label_sequence)
            node_map[node_label] = node_id
            nodes.append(f'{node_id}["{node_label} "]')
        else:
            node_id = node_map[node_label]

        if prev_node_id and verb_phrase:
            edges.append(f'{prev_node_id} -->|{verb_phrase}| {node_id}')
        elif prev_node_id:
            edges.append(f'{prev_node_id} --> {node_id}')

        prev_node_id = node_id

    last_label = next(label_sequence)
    nodes.append(f'{last_label}["End"]')
    edges.append(f'{prev_node_id} --> {last_label}')

    return "graph TD\n" + "\n".join(nodes) + "\n" + "\n".join(edges)

# Generate Mermaid.js code using LLM
def generate_mermaid_code(processed_inst):
    client = ollama.Client()
    prompt = f'''
    Given the following instruction set, generate the corresponding Mermaid.js linear flowchart code without any additional text or explanations.
    Instruction Set: {"\n".join(processed_inst)}
    '''

    response = client.generate(model="llama3.1", prompt=prompt, stream=True)
    mermaid_code = "".join(chunk["response"] for chunk in response)
    return mermaid_code

# Display Mermaid.js chart
def display_mermaid_chart(mermaid_code):
    graphbytes = mermaid_code.encode("utf8")
    base64_string = base64.urlsafe_b64encode(graphbytes).decode("ascii")
    image_url = f"https://mermaid.ink/img/{base64_string}"
    st.image(image_url)

# Center alignment CSS
center_alignment_css = """
<style>
    .main > div {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .element-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    .stButton button {
        margin-top: 10px;
    }
</style>
"""
st.markdown(center_alignment_css, unsafe_allow_html=True)

# Initialize Streamlit session state
if "selected_method" not in st.session_state:
    st.session_state.selected_method = None

# Streamlit UI
st.title("Instruction to Flowchart Generator")
st.markdown("### Upload an image containing instructions and generate a flowchart.")

uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Flowchart"):
        extracted_text = extract_text_from_image(image)
        st.subheader("Extracted Text")
        st.write(extracted_text)

        preprocessed_sentences = preprocess_sentences(extracted_text)

        st.subheader("Flowchart using POS tagging")
        with st.spinner("Flowchart is being generated..."):
            mermaid_code_method1 = parse_instructions(preprocessed_sentences)
        display_mermaid_chart(mermaid_code_method1)

        st.subheader("Flowchart Using Llama 3.1 model")
        with st.spinner("Generating flowchart using Method 2..."):
            mermaid_code_method2 = generate_mermaid_code(preprocessed_sentences)
        display_mermaid_chart(mermaid_code_method2)

        def data_entry(selected_method):
            data_entry = {
                "instruction": extracted_text,
                "method_1_output_code": mermaid_code_method1,
                "method_2_output_code": mermaid_code_method2,
                "human_eval": selected_method
            }
            try:
                with open("human_valuation.json", "r") as file:
                    data = json.load(file)
            except FileNotFoundError:
                data = []

            data.append(data_entry)

            with open("human_valuation.json", "w") as file:
                json.dump(data, file, indent=4)

        st.markdown("### Choose your preferred method")
        if st.button("Method 1", on_click=lambda: data_entry("method 1")):
            st.success("Selected method 1 successfully")
        if st.button("Method 2", on_click=lambda: data_entry("method 2")):
            st.success("Selected method 2 successfully")
