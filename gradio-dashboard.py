import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
import os

load_dotenv()

# --- Load book data ---
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# --- Embedding model ---
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Directory for persistent DB ---
PERSIST_DIR = "chroma_books_db"

if os.path.exists(PERSIST_DIR):
    print("✅ Loading existing Chroma DB...")
    db_books = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
else:
    print("⚡ Building new Chroma DB...")
    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(
        documents,
        embedding=embedding,
        persist_directory=PERSIST_DIR
    )
    db_books.persist()
    print("✅ Saved Chroma DB for future runs")

# --- Recommendation Logic ---
def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None,
                                      initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

# --- Options ---
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# --- Custom CSS for modern UI ---
custom_css = """
#main-container {
    max-width: 900px;
    margin: auto;
    padding: 2rem;
}
#header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(to right, #7f5af0, #2cb67d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
#header p {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
}
#search-card {
    background: #1e1e2f;
    padding: 2rem;
    border-radius: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
#search-button {
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white;
    font-weight: bold;
    border-radius: 0.75rem;
    padding: 0.75rem;
    transition: 0.3s;
}
#search-button:hover {
    opacity: 0.9;
}
"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as dashboard:
    with gr.Column(elem_id="main-container"):
        with gr.Row(elem_id="header"):
            gr.Markdown("<h1>Koutaiba Book Recomandar AI</h1><p>Your intelligent guide to the world of books.</p>")

        with gr.Column(elem_id="search-card"):
            user_query = gr.Textbox(
                placeholder="e.g., A thrilling space opera with political intrigue...",
                show_label=False
            )
            with gr.Row():
                tone_radio = gr.Radio(tones, label="Desired Tone", value="All")
                category_radio = gr.Radio(categories, label="Category", value="All")
            with gr.Row():
                clear_button = gr.Button("Clear", variant="secondary")
                search_button = gr.Button("Find My Next Book", elem_id="search-button")

        gr.Markdown("## Recommendations")
        output = gr.Gallery(columns=4, rows=2, object_fit="cover")

        # Events
        search_button.click(
            fn=recommend_books,
            inputs=[user_query, category_radio, tone_radio],
            outputs=output
        )
        user_query.submit(
            fn=recommend_books,
            inputs=[user_query, category_radio, tone_radio],
            outputs=output
        )
        clear_button.click(
            fn=lambda: ("", "All", "All", []),
            outputs=[user_query, category_radio, tone_radio, output]
        )



if __name__ == "__main__":
    dashboard.launch()

