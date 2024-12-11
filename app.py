# File path: app.py

import streamlit as st
import pytesseract
import spacy
import ollama

from PIL import Image
# from io import BytesIO
import json
# import requests
import base64
import string
import re


pytesseract.pytesseract.tesseract_cmd = r'd:\tesseract\tesseract.exe'
# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# OCR function
def extract_text_from_image(image):
    extracted_text = pytesseract.image_to_string(image)

    lines = extracted_text.splitlines()  # Split text by lines
        
        # Filter out blank lines and non-instructional lines
    instruction_lines = [
        re.sub(r"[^a-zA-Z,\s]", "", line.strip())        
        for line in lines 
        if line.strip()  # Include any line with content, not just numbered lines
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
            # Split on sentence-ending punctuation or coordinating conjunctions (like "and", "or", "but") 
            if token.dep_ in {"cc", "punct"} and token.text in {";",  "or", "but"}:
                # Append chunk as a sentence if it has meaningful content
                if current_chunk:
                    split_sentences.append(" ".join(current_chunk).strip())
                    current_chunk = []
        
        # Add any remaining content in the current chunk as a sentence
        if current_chunk:
            split_sentences.append(" ".join(current_chunk).strip())
        final_split=[re.sub(r'\s*,\s*', ',', sentence) for sentence in split_sentences]
    return final_split

# Parsing function to generate mermaid code using Method 1
def parse_instructions(sentences):
    nodes = ['A["Start"]']  # Start node
    edges = []
    node_map = {}
    node_counter = 0
    label_sequence = iter(string.ascii_uppercase)
    prev_node_id = next(label_sequence)
    
    # Generate labels A, B, C, etc.

    for sentence in sentences:
        doc = nlp(sentence)

        # Extract main verb and meaningful noun phrases
        verb_phrase = None
        noun_phrases = []

        # Extract verb (root of sentence) and any important modifiers
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                verb_phrase = token.lemma_.capitalize()
                for child in token.children:
                    if child.dep_ in {"prep", "advmod", "dobj"}:
                        verb_phrase += f" {child.text}"

        # Collect noun chunks and use full text as coherent noun phrases
        for chunk in doc.noun_chunks:
            noun_phrase = chunk.text.strip()
            noun_phrases.append(noun_phrase)

        # Merge noun phrases to form a coherent node label
        node_label = ", ".join(noun_phrases)

        # Map node labels to unique identifiers in alphabetical order
        if node_label not in node_map:
            node_id = next(label_sequence)
            node_map[node_label] = node_id
            nodes.append(f'{node_id}["{node_label} "]')
        else:
            node_id = node_map[node_label]

        # Create edges to link previous node with the current node based on the verb phrase
        
        if prev_node_id and verb_phrase:
            edges.append(f'{prev_node_id} -->|{verb_phrase}| {node_id}')
        elif prev_node_id:
            edges.append(f'{prev_node_id} -->{node_id}')

        # Set the current node as the previous for the next iteration
        prev_node_id = node_id

    # Link the last instruction node to the "End" node
    last_label=next(label_sequence)
    nodes.append(f'{last_label}["End"]')
    edges.append(f'{prev_node_id} --> {last_label}')

    # Combine nodes and edges to form the mermaid code
    mermaid_code = "graph TD\n" + "\n".join(nodes) + "\n" + "\n".join(edges)
    return mermaid_code

# LLM-based function to generate Mermaid code (Method 2)
def generate_mermaid_code(processed_inst):
    client = ollama.Client()

    # Define the prompt with the given instructions
    prompt = f'''
    Given the following instruction set, generate the corresponding Mermaid.js linear flowchart code  without any additional text or explanations.
    Only output the Mermaid.js code without any styles.

    For example if the input is:
        Fill a pot with potting soil
        Make a small hole in the soil
        Place the seed in the hole
        Cover the seed with soil
        Water gently
        Place the pot in a sunny location
    the output should be just:
        graph TD
            A["Start"]
            B[Fill pot with soil]
            C[Make hole in soil]
            D[Place seed in hole]
            E[Cover seed with soil]
            F[Water gently]
            G[Place in sunny location]
            H[End]
            A --> B
            B --> C
            C --> D
            D --> E
            E --> F
            F --> G
            G --> H

    Instruction Set: {"\n".join(processed_inst)}
    '''

    # Generate a streaming response
    response = client.generate(model="llama3.1", prompt=prompt, stream=True)

    # Process the response as it arrives and accumulate the code
    mermaid_code = ""
    # st.write_stream(i["response"] for i in response)
    for chunk in response:
        mermaid_code += chunk["response"]
        # st.write(chunk["response"],end="")
    return mermaid_code

# Function to render Mermaid chart
def display_mermaid_chart(mermaid_code):
    graphbytes = mermaid_code.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    image_url = f"https://mermaid.ink/img/{base64_string}"
    st.image(image_url)



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
if "selected_method" not in st.session_state:
    st.session_state.selected_method = None
# Streamlit UI
st.title("Instruction to Flowchart Generator")
st.markdown("### Upload an image containing instructions and generate a flowchart.")

# Image upload
uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Generate Flowchart"):
        # Extract text using OCR
        extracted_text = extract_text_from_image(image)
        st.subheader("Extracted Text")
        st.write(extracted_text)
        
        # Preprocess text for flowchart generation
        # instruction_lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
        preprocessed_sentences = preprocess_sentences(extracted_text)
        
        # Generate Mermaid code using Method 1
        st.subheader("Flowchart using POS tagging")
        with st.spinner("Flowchart is being generated..."):
            mermaid_code_method1 = parse_instructions(preprocessed_sentences)
        display_mermaid_chart(mermaid_code_method1)

        # Generate Mermaid code using Method 2 (LLM-based)
        # mermaid_code_method2 = generate_mermaid_code(preprocessed_sentences)
        st.subheader("Flowchart Using LLama 3.1 model")
        with st.spinner("Generating flowchart using Method 2..."):
            mermaid_code_method2 = generate_mermaid_code(preprocessed_sentences)
        # st.success("Method 2 Flowchart Generated!")        
        display_mermaid_chart(mermaid_code_method2)

        if "selected_method" not in st.session_state:
            st.session_state.selected_method = None
            st.session_state.selection_made = False  # Tracks if the user has made a choice

        def data_entry(selected_method):
            data_entry = {
                    "instruction": extracted_text,
                    "method_1_output_code": mermaid_code_method1,
                    "method_2_output_code": mermaid_code_method2,
                    "human_eval": selected_method
                }

                # Write to JSON file
            try:
                with open("human_valuation.json", "r") as file:
                    data = json.load(file)
            except FileNotFoundError:
                data = []

            data.append(data_entry)

            with open("human_valuation.json", "w") as file:
                json.dump(data, file, indent=4)



        st.markdown("### Choose your preferred method")
        # selected_method = None
        if st.button("Method 1",on_click=data_entry("method 1")):
            st.success("Selected method 1 successfully")
            st.button("done")
        if st.button("Method 2",on_click=data_entry("method 2")):
            st.success("Selected method 2 successfully")
            st.button("done")

        # selected_method = st.radio("Select your preferred flowchart output:", ("Method 1", "Method 2"))
        # if selected_method:
        #     st.session_state.selection_made = True
        # # Save selection to JSON
        # if st.session_state.selection_made and st.button("Save Selection"):
        #     data_entry = {
        #         "instruction": extracted_text,
        #         "method_1_output_code": mermaid_code_method1,
        #         "method_2_output_code": mermaid_code_method2,
        #         "human_eval": selected_method
        #     }

        #     # Write to JSON file
        #     try:
        #         with open("human_valuation.json", "r") as file:
        #             data = json.load(file)
        #     except FileNotFoundError:
        #         data = []

        #     data.append(data_entry)

        #     with open("human_valuation.json", "w") as file:
        #         json.dump(data, file, indent=4)
            

        #     st.success(f"Your selection '{selected_method}' has been saved.")
        # else:
        #     st.warning("Please make a selection and click 'Save Selection'.")