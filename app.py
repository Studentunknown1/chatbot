from flask import Flask, request, render_template_string
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

df = pd.read_csv("attractive_pickup_lines_dataset.csv")

pickup_lines_by_lang = {
    lang: df[df['language'] == lang]['pickup_line'].tolist()
    for lang in df['language'].unique()
}

model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_indexes = {}

for lang, lines in pickup_lines_by_lang.items():
    embeddings = model.encode(lines, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss_indexes[lang] = {
        'index': index,
        'lines': lines
    }

app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <title>Flirty Chatbot (PK & EN)</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 30px; background-color: #e6f2e6; }
        h1 { color: #1b5e20; }
        form { margin-top: 20px; }
        input, select { padding: 10px; margin: 5px 0; width: 100%; }
        .chatbox { border: 1px solid #ccc; padding: 15px; background: white; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Flirty Chatbot 💚 (English + Pakistani)</h1>
    <div class=\"chatbox\">
        <p><strong>Bot:</strong> Assalamualaikum janeman 😍! Choose your language and start flirting...</p>
        {% if pickup_line %}
            <p><strong>You:</strong> {{ user_input }}</p>
            <p><strong>Bot:</strong> {{ pickup_line }}</p>
        {% endif %}
        <form method=\"post\">
            <label for=\"language\">Select Language:</label>
            <select name=\"language\" required>
                <option value=\"english\">English</option>
                <option value=\"pakistani\">Pakistani</option>
            </select>
            <label for=\"user_input\">Say something 💬:</label>
            <input type=\"text\" name=\"user_input\" required>
            <button type=\"submit\">Flirt 😘</button>
        </form>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def chat():
    pickup_line = None
    user_input = None

    if request.method == 'POST':
        user_input = request.form['user_input']
        language = request.form['language']

        if language in faiss_indexes:
            user_embedding = model.encode([user_input], convert_to_numpy=True)
            faiss_index = faiss_indexes[language]['index']
            lines = faiss_indexes[language]['lines']
            distances, indices = faiss_index.search(user_embedding, 1)
            best_index = indices[0][0]
            pickup_line = lines[best_index]

    return render_template_string(html_template, pickup_line=pickup_line, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
