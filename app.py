import os
import streamlit as st
from openai import OpenAI
import base64
from utils import get_image_description
from sentence_transformers import SentenceTransformer, util

# Load the sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to calculate semantic similarity between two texts
def calculate_semantic_similarity(text1, text2):
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    return similarity.item()

# Streamlit app layout
st.title("Image Relevance to News Text using GPT-4o")
st.write("Upload a news text and images to find out which images are relevant to the text.")

# Textbox for updating OpenAI API key
api_key = st.text_input("Enter your OpenAI API key", type="password")
if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY", "")

if api_key:
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    # Text area for news text input
    news_text = st.text_area("Enter the news text")

    # Upload multiple images button
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if news_text and uploaded_files:
        # Store descriptions and similarity scores
        image_relevancies = []

        for uploaded_file in uploaded_files:
            try:
                # Display the uploaded image
                st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
                st.write("Classifying...")

                # Get the image description
                description = get_image_description(client, uploaded_file, prompt="Describe the image.")
                st.write(description)

                # Calculate semantic similarity between news text and image description
                similarity = calculate_semantic_similarity(news_text, description)
                image_relevancies.append({
                    'similarity': similarity,
                    'image': uploaded_file,
                    'description': description
                })
            except Exception as e:
                st.error(f"Error: {e}")

        # Sort images by relevance
        image_relevancies = sorted(image_relevancies, key=lambda x: x['similarity'], reverse=True)

        st.write("Relevant Images:")
        for item in image_relevancies:
            st.image(item['image'], caption=f"Description: {item['description']} (Similarity: {item['similarity']:.2f})", use_column_width=True)
else:
    st.error("Please provide a valid OpenAI API key.")

# Function to get image description
def get_image_description(client, uploaded_file, prompt):
    # Encode the uploaded image in base64
    encoded_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

    # Create the GPT-4o API request
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    # Extract and return the description
    return response.choices[0].message.content
