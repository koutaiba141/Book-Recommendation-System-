# Books LLM - Semantic Book Recommendation System

A comprehensive book recommendation system that combines machine learning, natural language processing, and vector similarity search to provide personalized book recommendations based on semantic understanding of book descriptions and emotional analysis.

## 🌟 Features

- **Semantic Search**: Vector-based similarity search using HuggingFace embeddings
- **Emotion Analysis**: Books are classified by emotional tones (joy, sadness, anger, fear, surprise, disgust)
- **Category Filtering**: Filter recommendations by simplified book categories (Fiction/Nonfiction)
- **Interactive Dashboard**: Gradio-powered web interface for easy interaction
- **Rich Dataset**: 5,197+ books with metadata, descriptions, and emotional scores

## 🔧 System Architecture

The system consists of four main components:

1. **Data Exploration** (`data-exploration.ipynb`): Data cleaning, preprocessing, and analysis
2. **Text Classification** (`text-classification.ipynb`): Category simplification and emotion analysis
3. **Vector Search** (`vector-search.ipynb`): Embedding generation and vector database creation
4. **Web Dashboard** (`gradio-dashboard.py`): Interactive recommendation interface

## 📊 Dataset

The project uses the [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) from Kaggle, containing:

- Book titles, authors, descriptions
- Publication years, ratings, page counts
- Google Books thumbnails
- ISBN-10 and ISBN-13 identifiers

### Processing Pipeline

1. **Data Cleaning**: Remove books with insufficient descriptions (<25 words)
2. **Category Simplification**: Map 479+ categories to Fiction/Nonfiction
3. **Emotion Analysis**: Extract emotional scores from book descriptions
4. **Text Preprocessing**: Create tagged descriptions for vector search

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd books_llm
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (create `.env` file):
```
# Add any required API keys or configuration
```

### Running the Application

1. **Start the Gradio Dashboard**:
```bash
python gradio-dashboard.py
```

2. Open your browser and navigate to the displayed URL (typically `http://localhost:7860`)

3. Enter a book description, select category and emotional tone preferences, and get personalized recommendations!

## 📁 File Structure

```
books_llm/
├── data-exploration.ipynb      # Data analysis and cleaning
├── text-classification.ipynb   # Category and emotion processing
├── vector-search.ipynb        # Vector embedding and search setup
├── gradio-dashboard.py        # Interactive web interface
├── books_cleaned.csv          # Processed book dataset
├── books_with_categories.csv  # Books with simplified categories
├── books_with_emotions.csv    # Books with emotion scores
├── tagged_description.txt     # Preprocessed text for embeddings
├── cover_not.png             # Default book cover image
└── requirements.txt          # Python dependencies
```

## 🎯 How It Works

### 1. Semantic Search
- Uses sentence-transformers/all-MiniLM-L6-v2 for text embeddings
- ChromaDB vector database for efficient similarity search
- Searches through book descriptions to find semantically similar content

### 2. Recommendation Algorithm
```python
def retrieve_semantic_recommendations(query, category, tone, initial_top_k=50, final_top_k=16):
    # 1. Semantic similarity search
    recs = db_books.similarity_search(query, k=initial_top_k)
    
    # 2. Category filtering
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]
    
    # 3. Emotional tone sorting
    if tone != "All":
        book_recs.sort_values(by=emotion_mapping[tone], ascending=False)
```

### 3. Emotional Analysis
Books are scored across seven emotional dimensions:
- **Joy**: Uplifting, positive content
- **Sadness**: Melancholic, emotional themes
- **Anger**: Intense, conflict-driven narratives
- **Fear**: Suspenseful, thriller elements
- **Surprise**: Unexpected plot twists
- **Disgust**: Dark, uncomfortable themes
- **Neutral**: Balanced emotional tone

## 🔍 Usage Examples

### Basic Search
```
Query: "A story about friendship and overcoming challenges"
Category: All
Tone: Happy
```

### Specific Genre Search
```
Query: "Mystery with detective solving crimes"
Category: Fiction
Tone: Suspenseful
```

### Emotional Preference
```
Query: "Coming of age story"
Category: Fiction
Tone: Sad
```

## 🛠️ Development

### Running Jupyter Notebooks

1. **Data Exploration**: Understand dataset structure and quality
2. **Text Classification**: Process categories and emotions
3. **Vector Search**: Set up embeddings and test similarity search

### Customization

- **Add new emotions**: Modify emotion analysis in `text-classification.ipynb`
- **Change embedding model**: Update model in `gradio-dashboard.py`
- **Adjust recommendation logic**: Modify `retrieve_semantic_recommendations()`

## 📈 Performance

- **Dataset Size**: 5,197 books
- **Search Speed**: ~100ms per query
- **Embedding Model**: 384-dimensional vectors
- **Memory Usage**: ~500MB for vector database

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Kaggle](https://kaggle.com) for the books dataset
- [HuggingFace](https://huggingface.co) for transformer models
- [LangChain](https://langchain.com) for vector search utilities
- [Gradio](https://gradio.app) for the web interface framework

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing documentation in notebooks
- Review the dataset source for data-related questions

---

*Built with ❤️ for book lovers and recommendation system enthusiasts*
