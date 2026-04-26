WELCOME TO: FLORASENSE 


# 🌸 Flower Recommendation & Visualization Platform

## Overview

This project is an intelligent flower recommendation and visualization system that combines traditional information retrieval techniques with modern AI-powered reasoning. It allows users to search for flowers based on meaning, occasion, aesthetics, and care preferences, and returns personalized recommendations with rich explanations.

The system integrates:

* Classical recommendation methods (SVD, TF-IDF)
* Retrieval-Augmented Generation (RAG)
* Optional Large Language Model (LLM) reasoning
* Interactive frontend visualization

---

## 🚀 Key Features

### 1. Smart Flower Recommendations

* Users can input natural language queries (e.g., *"flowers for gratitude"*, *"low maintenance romantic flowers"*)
* The system extracts meaningful attributes like:

  * Occasion
  * Symbolism
  * Color
  * Maintenance level
* Returns ranked flower suggestions with relevance scores

### 2. Dual Recommendation Engine

* **SVD-based recommendations**: captures latent relationships between flowers and attributes
* **TF-IDF fallback**: ensures robust performance even when SVD fails

### 3. Retrieval-Augmented Generation (RAG)

* Transforms user queries into optimized search queries
* Retrieves structured flower data
* Uses LLMs (optional) to generate:

  * Natural language summaries
  * "Why this matches" explanations
  * Occasion-based recommendations

### 4. AI-Powered Explanations

* When enabled, the system:

  * Rewrites explanations into polished, human-friendly text
  * Improves readability of raw data
  * Generates comparative summaries across flowers

### 5. Hard Filtering System

* Supports exclusion constraints (e.g., *"not roses", "not yellow"*)
* Filters results strictly based on user intent

### 6. Flower Visualization

* Interactive visualization of flower sets
* Bouquet-level insights:

  * Combined meanings
  * Suggested complementary flowers

### 7. Autocomplete Search

* Provides real-time query suggestions
* Helps users refine searches quickly

---

### Backend (Flask)

* REST API endpoints for:

  * Search (`/api/recommendations`)
  * RAG recommendations (`/api/rag-recommendations`)
  * Autocomplete (`/api/autocomplete`)
  * Visualization data (`/api/visualizer-flowers`)
  
* Modular design with:

  * Recommendation engines
  * LLM integration layer
  * Data normalization and filtering

### Frontend (React + Vite)

* Interactive UI for:

  * Searching and browsing flowers
  * Viewing explanations and summaries
  * Visualizing results and building your own bouquet 

### Data Layer

* Flower dataset includes:

  * Names and scientific names
  * Meanings and symbolism
  * Occasions
  * Colors
  * Maintenance requirements

---

## 🔄 How It Works!

1. **User Query Input**

   * Natural language input (e.g., "white flower for gratitude")

2. **Query Processing**

   * Optional LLM transforms query into structured search terms
   * Extracts exclusions and intent

3. **Retrieval Phase**

   * SVD or TF-IDF retrieves relevant flowers
   * Results are filtered based on constraints

4. **RAG Context Building**

   * Top results are converted into structured context documents

5. **Generation Phase (Optional LLM)**

   * Produces:
     * Overall recommendation summary
     * Per-flower explanations
     * Occasion-specific insights

6. **Response Assembly**

   * Combines raw data + generated text
   * Returns structured JSON to frontend
    
7. **3D Visualizer!**
   * See how the flowers connect to each other
   * Build your OWN bouquet and see what meanings it captures!
   * Receive suggestions for additions to the bouquet
---

## 🧪 Example Queries

* "romantic flowers that are easy to take care of"
* "flowers to say thank you"
* "sympathy flowers but not lilies"
* "red flowers for celebrations"

---

## How to run
### Windows
```bash
# 1. Set up Python virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start Flask backend (in one terminal)
python src/app.py

# 4. In a NEW terminal, install and start React
cd frontend
npm install
npm run dev
```

### Mac/Linux
```bash
# 1. Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start Flask backend (in one terminal)
python src/app.py

# 4. In a NEW terminal, install and start React
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` in your browser!

