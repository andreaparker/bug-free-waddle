import os
import uuid
import json
import time
import shutil
import markdown
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from markupsafe import Markup
from werkzeug.utils import secure_filename
from logger import get_logger
from byaldi import RAGMultiModalModel
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

logger = get_logger(__name__)
logger.info("Application started.")

# Configuration
app.config.update(
    UPLOAD_FOLDER='uploaded_documents',
    STATIC_FOLDER='static',
    SESSION_FOLDER='sessions',
    INDEX_FOLDER=os.path.join(os.getcwd(), '.byaldi'),  # Set to .byaldi folder in current directory
    INITIALIZATION_DONE=False,
    AVAILABLE_INDEXER_MODELS=['vidore/colpali', 'vidore/colpali-v1.2'],
    AVAILABLE_GENERATION_MODELS=['your_generation_model']  # Replace with your actual model
)

# Create necessary directories if they don't exist
for folder in ['UPLOAD_FOLDER', 'STATIC_FOLDER', 'SESSION_FOLDER', 'INDEX_FOLDER']:
    os.makedirs(app.config[folder], exist_ok=True)

# Global dictionary to store RAG models per session
RAG_models = {}


def load_rag_model_for_session(session_id):
    """
    Loads the RAG model for the given session_id from the index on disk.
    """
    index_path = os.path.join(app.config['INDEX_FOLDER'], session_id)

    if os.path.exists(index_path):
        try:
            RAG = RAGMultiModalModel.from_index(index_path)
            RAG_models[session_id] = RAG
            logger.info(f"RAG model for session {session_id} loaded from index.")
        except Exception as e:
            logger.error(f"Error loading RAG model for session {session_id}: {e}")
    else:
        logger.warning(f"No index found for session {session_id}.")


def load_existing_indexes():
    """
    Loads all existing indexes from the .byaldi folder when the application starts.
    """
    if os.path.exists(app.config['INDEX_FOLDER']):
        for session_id in os.listdir(app.config['INDEX_FOLDER']):
            if os.path.isdir(os.path.join(app.config['INDEX_FOLDER'], session_id)):
                load_rag_model_for_session(session_id)
    else:
        logger.warning("No .byaldi folder found. No existing indexes to load.")


@app.before_request
def initialize_app():
    """
    Initializes the application by loading existing indexes.
    This will run before the first request, but only once.
    """
    if not app.config['INITIALIZATION_DONE']:
        load_existing_indexes()
        app.config['INITIALIZATION_DONE'] = True
        logger.info("Application initialized and indexes loaded.")


@app.before_request
def make_session_permanent():
    session.permanent = True
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())


def get_session_data(session_id):
    """
    Retrieves session data from the session file.
    """
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            return json.load(f)
    else:
        return {
            'chat_history': [],
            'session_name': 'Untitled Session',
            'indexed_files': []
        }


def save_session_data(session_id, session_data):
    """
    Saves session data to the session file.
    """
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")
    with open(session_file, 'w') as f:
        json.dump(session_data, f)


def get_available_sessions():
    """
    Retrieves a list of available chat sessions.
    """
    session_files = os.listdir(app.config['SESSION_FOLDER'])
    chat_sessions = []
    for file in session_files:
        if file.endswith('.json'):
            s_id = file[:-5]
            data = get_session_data(s_id)
            name = data.get('session_name', 'Untitled Session')
            chat_sessions.append({'id': s_id, 'name': name})
    return chat_sessions


@app.route('/', methods=['GET'])
def home():
    return redirect(url_for('chat'))


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    session_id = session['session_id']
    session_data = get_session_data(session_id)
    chat_history = session_data.get('chat_history', [])
    session_name = session_data.get('session_name', 'Untitled Session')
    indexed_files = session_data.get('indexed_files', [])

    if request.method == 'POST':
        if 'upload' in request.form:
            return handle_file_upload(session_id, session_data)
        elif 'send_query' in request.form:
            return handle_send_query(session_id, session_data)

    # For GET requests, render the chat page
    chat_sessions = get_available_sessions()
    indexer_model = session.get('indexer_model', app.config['AVAILABLE_INDEXER_MODELS'][0])
    generation_model = session.get('generation_model', app.config['AVAILABLE_GENERATION_MODELS'][0])
    resized_height = session.get('resized_height', 280)
    resized_width = session.get('resized_width', 280)

    return render_template(
        'chat.html',
        chat_history=chat_history,
        chat_sessions=chat_sessions,
        current_session=session_id,
        indexer_model=indexer_model,
        generation_model=generation_model,
        resized_height=resized_height,
        resized_width=resized_width,
        session_name=session_name,
        indexed_files=indexed_files
    )


def handle_file_upload(session_id, session_data):
    """
    Handles file upload and indexing.
    """
    files = request.files.getlist('file')
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_folder, exist_ok=True)
    uploaded_files = []

    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(session_folder, filename)
            file.save(file_path)
            uploaded_files.append(filename)
            logger.info(f"File saved: {file_path}")

    if uploaded_files:
        try:
            index_name = session_id
            index_path = os.path.join(app.config['INDEX_FOLDER'], index_name)
            indexer_model = session.get('indexer_model', app.config['AVAILABLE_INDEXER_MODELS'][0])
            RAG = index_documents(session_folder, index_name=index_name, index_path=index_path, indexer_model=indexer_model)
            if RAG is None:
                raise ValueError("Indexing failed: RAG model is None")
            RAG_models[session_id] = RAG
            session['index_name'] = index_name
            session['session_folder'] = session_folder
            session_data['indexed_files'].extend(uploaded_files)
            # Store the indexer model in session data
            session_data['indexer_model'] = indexer_model
            save_session_data(session_id, session_data)
            logger.info("Documents indexed successfully.")
            return jsonify({
                "success": True,
                "message": "Files indexed successfully.",
                "indexed_files": session_data['indexed_files']
            })
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return jsonify({"success": False, "message": f"Error indexing files: {str(e)}"})
    else:
        return jsonify({"success": False, "message": "No files were uploaded."})


def handle_send_query(session_id, session_data):
    """
    Handles the user's query input.
    """
    query = request.form['query']
    try:
        generation_model = session.get('generation_model', app.config['AVAILABLE_GENERATION_MODELS'][0])
        resized_height = session.get('resized_height', 280)
        resized_width = session.get('resized_width', 280)

        # Retrieve the indexer model from session_data
        indexer_model = session_data.get('indexer_model', session.get('indexer_model', app.config['AVAILABLE_INDEXER_MODELS'][0]))

        # Retrieve relevant documents
        rag_model = RAG_models.get(session_id)
        if rag_model is None:
            logger.error(f"RAG model not found for session {session_id}")
            return jsonify({"success": False, "message": "RAG model not found for this session."})

        retrieved_images = retrieve_documents(rag_model, query, session_id)
        logger.info(f"Retrieved images: {retrieved_images}")

        # Generate response with full image paths
        full_image_paths = [os.path.join(app.static_folder, img) for img in retrieved_images]
        response = generate_response(
            full_image_paths, query, session_id, resized_height, resized_width, generation_model
        )

        # Parse markdown in the response
        parsed_response = Markup(markdown.markdown(response))

        # Update chat history with indexer model
        chat_history = session_data.get('chat_history', [])
        chat_history.append({"role": "user", "content": query})
        chat_history.append({
            "role": "assistant",
            "content": parsed_response,
            "images": retrieved_images,
            "indexer_model": indexer_model  # Include the indexer model here
        })
        session_data['chat_history'] = chat_history

        # Update session name if it's the first message
        if len(chat_history) == 2:  # First user message and AI response
            session_data['session_name'] = query[:50]  # Truncate to 50 characters

        save_session_data(session_id, session_data)

        # Render the new messages
        new_messages_html = render_template('chat_messages.html', messages=[
            {"role": "user", "content": query},
            {"role": "assistant", "content": parsed_response, "images": retrieved_images, "indexer_model": indexer_model}
        ])

        return jsonify({
            "success": True,
            "html": new_messages_html
        })
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return jsonify({"success": False, "message": f"An error occurred while generating the response: {str(e)}"})


@app.route('/switch_session/<session_id>')
def switch_session(session_id):
    session['session_id'] = session_id
    if session_id not in RAG_models:
        load_rag_model_for_session(session_id)
    flash("Switched to session.", "info")
    return redirect(url_for('chat'))


@app.route('/rename_session', methods=['POST'])
def rename_session():
    session_id = request.form.get('session_id')
    new_session_name = request.form.get('new_session_name', 'Untitled Session')
    session_data = get_session_data(session_id)
    session_data['session_name'] = new_session_name
    save_session_data(session_id, session_data)
    return jsonify({"success": True, "message": "Session name updated."})


@app.route('/delete_session/<session_id>', methods=['POST'])
def delete_session(session_id):
    try:
        # Delete session file
        session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)

        # Delete uploaded files
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(session_folder):
            shutil.rmtree(session_folder)

        # Delete session images
        session_images_folder = os.path.join(app.config['STATIC_FOLDER'], 'images', session_id)
        if os.path.exists(session_images_folder):
            shutil.rmtree(session_images_folder)

        # Remove RAG model from memory
        RAG_models.pop(session_id, None)

        # Reset session if current session is deleted
        if session.get('session_id') == session_id:
            session['session_id'] = str(uuid.uuid4())

        logger.info(f"Session {session_id} deleted.")
        return jsonify({"success": True, "message": "Session deleted successfully."})
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return jsonify({"success": False, "message": f"An error occurred while deleting the session: {str(e)}"})


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        indexer_model = request.form.get('indexer_model', app.config['AVAILABLE_INDEXER_MODELS'][0])
        generation_model = request.form.get('generation_model', app.config['AVAILABLE_GENERATION_MODELS'][0])
        resized_height = int(request.form.get('resized_height', 280))
        resized_width = int(request.form.get('resized_width', 280))

        session['indexer_model'] = indexer_model
        session['generation_model'] = generation_model
        session['resized_height'] = resized_height
        session['resized_width'] = resized_width
        session.modified = True

        logger.info(f"Settings updated: indexer_model={indexer_model}, generation_model={generation_model}, "
                    f"resized_height={resized_height}, resized_width={resized_width}")
        flash("Settings updated.", "success")
        return redirect(url_for('chat'))
    else:
        indexer_model = session.get('indexer_model', app.config['AVAILABLE_INDEXER_MODELS'][0])
        generation_model = session.get('generation_model', app.config['AVAILABLE_GENERATION_MODELS'][0])
        resized_height = session.get('resized_height', 280)
        resized_width = session.get('resized_width', 280)
        return render_template(
            'settings.html',
            indexer_model=indexer_model,
            generation_model=generation_model,
            resized_height=resized_height,
            resized_width=resized_width,
            available_indexer_models=app.config['AVAILABLE_INDEXER_MODELS'],
            available_generation_models=app.config['AVAILABLE_GENERATION_MODELS']
        )


@app.route('/new_session')
def new_session():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    session_files = os.listdir(app.config['SESSION_FOLDER'])
    session_number = len([f for f in session_files if f.endswith('.json')]) + 1
    session_name = f"Session {session_number}"
    session_data = {
        'session_name': session_name,
        'chat_history': [],
        'indexed_files': []
    }
    save_session_data(session_id, session_data)
    flash("New chat session started.", "success")
    return redirect(url_for('chat'))


@app.route('/get_indexed_files/<session_id>')
def get_indexed_files(session_id):
    session_data = get_session_data(session_id)
    indexed_files = session_data.get('indexed_files', [])
    return jsonify({"success": True, "indexed_files": indexed_files})


if __name__ == '__main__':
    app.run(port=5050, debug=True)
