
# **Keyword Extraction from Text**

## **Project Overview**
This project is a Flask-based web application for extracting keywords from a given text passage using various methods, including TF-IDF, Jaccard Similarity, and Co-occurrence. Users can input text and select the desired method for keyword extraction, and the application will display the extracted keywords.

## **Features**
- **Multiple Extraction Methods:** Users can choose between TF-IDF, Jaccard, and Co-occurrence methods for keyword extraction.
- **User Input:** Accepts text input from the user via a web interface.
- **Dynamic Results:** Displays the extracted keywords based on the selected method.

## **Technologies Used**
- **Backend:** Python, Flask
- **Frontend:** HTML, CSS
- **NLP Libraries:** NLTK, Scikit-learn, ROUGE

## **Project Structure**
- `app.py`: Main Flask application file.
- `templates/`: Contains HTML files for the web pages.
  - `index.html`: Home page where users input text and select extraction method.
  - `result.html`: Page that displays the extracted keywords.
- `static/`: Contains static files like CSS.
  - `style.css`: Custom styles for the web pages.

## **Setup and Usage**
1. **Install Dependencies:**
   ```bash
   pip install flask nltk scikit-learn rouge-score
   ```
2. **Run the Application:**
   ```bash
   python app.py
   ```
3. **Access the Web App:**
   Open your web browser and go to `http://127.0.0.1:5000/`.

## **Future Enhancements**
- Add more keyword extraction methods.
- Improve the UI/UX of the application.
- Implement advanced preprocessing techniques.

---
