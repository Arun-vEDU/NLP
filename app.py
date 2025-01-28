import dash
from dash import dcc, html, Input, Output, State
from gensim.models import KeyedVectors
import numpy as np

# Load your trained Word2Vec model
model_path = "word_vectors.kv"  #  model path
word_vectors = KeyedVectors.load(model_path)

# Function to compute dot product similarity
def compute_dot_product(query, model, top_n=10):
    if query not in model.key_to_index:
        return None, f"The word '{query}' is not in the vocabulary!"

    query_vector = model[query]
    similarities = {}

    for word in model.index_to_key:
        if word != query:  # Avoid comparing the word with itself
            word_vector = model[word]
            dot_product = np.dot(query_vector, word_vector)
            similarities[word] = dot_product

    # Sort by similarity score and return the top N results
    top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_similar, None

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div(
    style={"fontFamily": "Arial", "maxWidth": "600px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H1("Word Similarity Finder", style={"textAlign": "center", "color": "#333"}),
        dcc.Input(
            id="query-word-input",
            type="text",
            placeholder="Enter a word...",
            style={"width": "100%", "padding": "10px", "marginBottom": "10px", "borderRadius": "4px", "border": "1px solid #ccc"},
        ),
        html.Button(
            "Find Similar Words",
            id="submit-button",
            n_clicks=0,
            style={"padding": "10px 15px", "background": "#007BFF", "color": "white", "border": "none", "borderRadius": "4px", "cursor": "pointer"},
        ),
        html.Div(id="output-container", style={"marginTop": "20px"}),
    ],
)

# Callback to handle word similarity computation
@app.callback(
    Output("output-container", "children"),
    Input("submit-button", "n_clicks"),
    State("query-word-input", "value"),
)
def update_output(n_clicks, query_word):
    if n_clicks > 0:
        if not query_word:
            return html.Div("Please enter a word.", style={"color": "red"})

        # Compute similar words
        top_similar_words, error_message = compute_dot_product(query_word, word_vectors)

        if error_message:
            return html.Div(error_message, style={"color": "red"})

        # Display the top similar words
        return html.Div(
            children=[
                html.H2(f"Top 10 Similar Words to '{query_word}'", style={"color": "#007BFF"}),
                html.Ul(
                    [html.Li(f"{word}: {score:.4f}") for word, score in top_similar_words],
                    style={"listStyleType": "none", "padding": "0", "color": "#333"},
                ),
            ]
        )
    return ""

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
