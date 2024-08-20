import torch
from PIL import Image
from flask import Flask, request, render_template, jsonify, session, send_from_directory
from sentence_transformers import SentenceTransformer
import openai
import chromadb
from chromadb.config import Settings
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
import os
import uuid

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = os.urandom(24)  # For session management

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ChromaDB client
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chromadb_data"
))
collection_name = "new_pdf_text_collection"
collection = client.get_collection(name=collection_name)

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Initialize OpenAI client with API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def process_image_with_clip(file_path):
    image = Image.open(file_path)
    inputs = clip_processor(
        images=image,
        text=["a photo of a house", "a photo of a plumbing issue",
              "a photo of an electrical problem", "a photo of a roof", "a photo of an HVAC system"],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = clip_model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)
    best_match = clip_processor.tokenizer.decode(
        inputs.input_ids[probs.argmax()])

    return f"The image appears to show {best_match}."


def get_rag_response(user_query, conversation_history, image_context=""):
    query_embedding = model.encode(user_query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    chroma_context = "\n".join(results['documents'][0])

    messages = [
        {"role": "system", "content": "You are a helpful assistant with expertise in home improvement, repairs, and maintenance. You have access to image descriptions and additional context. Use this information to answer questions."}
    ]

    # Add last 3 prompts from conversation history
    messages.extend(conversation_history[-3:])

    # Add current context and query
    if image_context:
        messages.append(
            {"role": "user", "content": f"An image has been uploaded. Image description: {image_context}"})
    messages.append(
        {"role": "user", "content": f"Additional Context:\n{chroma_context}\n\nUser's question: {user_query}"})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=300,
        n=1,
        temperature=0.7
    )

    return response['choices'][0]['message']['content'].strip()


@app.route('/')
def index():
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
        session['conversation_history'] = []
    return render_template('index.html', conversation_history=session['conversation_history'])


@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form['query']
    image_context = ""
    file_path = None

    if 'image' in request.files:
        image_file = request.files['image']
        if image_file.filename != '':
            filename = f"{session['conversation_id']}_{image_file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(file_path)
            image_context = process_image_with_clip(file_path)
            file_path = f"/uploads/{filename}"  # Update file_path to be a URL

    conversation_history = session.get('conversation_history', [])

    # If there's an image, add its context to the conversation history
    if image_context:
        conversation_history.append(
            {"role": "system", "content": f"An image has been uploaded. Image description: {image_context}"})

    answer = get_rag_response(user_query, conversation_history, image_context)

    conversation_history.append({"role": "user", "content": user_query})
    conversation_history.append({"role": "assistant", "content": answer})
    # Keep last 3 prompts (6 messages: 3 user, 3 assistant)
    session['conversation_history'] = conversation_history[-6:]

    return jsonify({
        "query": user_query,
        "answer": answer,
        "image_path": file_path
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
