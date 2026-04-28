# 🎬 Movie Recommendation System

A deep learning-based Movie Recommendation System built with 
PyTorch and Collaborative Filtering techniques, trained on the 
MovieLens dataset.

## 📌 Project Overview

This project implements an end-to-end recommendation system that 
predicts movie ratings for users based on their historical 
preferences. The model learns latent representations of users and 
movies through embedding layers and fully connected neural networks.

**Test RMSE: 0.89** — the model predicts ratings with an average 
error of less than 1 star.

## 📊 Dataset

- **Source:** MovieLens Latest Small Dataset (GroupLens)
- **Total ratings:** 100,836
- **Total users:** 610
- **Total movies:** 9,724
- **Rating range:** 0.5 to 5.0
- **Average rating:** 3.50

### Key Findings from Exploratory Data Analysis
- Ratings are positively skewed — users tend to rate movies 
they enjoyed
- Most common rating: 4.0 (over 26,000 ratings)
- Most rated movie: Forrest Gump (1994) with 329 ratings
- Highest rated movie: The Shawshank Redemption (1994) 
with 4.43 average
- 18 movies in the dataset have never been rated

## 🧠 Model Architecture

The model uses Neural Collaborative Filtering with the 
following architecture:

- **User Embedding:** 610 users → 50 dimensions
- **Movie Embedding:** 9,724 movies → 50 dimensions
- **Fully Connected Layers:** 100 → 128 → 64 → 1
- **Activation:** ReLU
- **Regularization:** Dropout (0.2)
- **Loss Function:** MSE
- **Optimizer:** Adam (lr=0.001)

## 📈 Results

| Metric | Value |
|--------|-------|
| Test MSE | 0.8099 |
| Test RMSE | 0.8999 |
| Training Epochs | 10 |
| Batch Size | 256 |

The training loss decreased from **1.53 to 0.70** across 
10 epochs, demonstrating effective learning with no signs 
of overfitting.

## 🛠️ Technologies

- Python 3.x
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

## 🚀 How to Run

1. Clone the repository
2. Open Movie_Recommendation_System.ipynb in Google Colab
3. Run all cells in order
4. Use the recommend_movies(user_id) function to get 
personalized recommendations

## 📁 Project Structure

| File | Description |
|------|-------------|
| Movie_Recommendation_System.ipynb | Main notebook with full pipeline |
| README.md | Project documentation |
| ml-latest-small/movies.csv | Movies dataset |
| ml-latest-small/ratings.csv | Ratings dataset |

## 💡 Example Output

```python
recommend_movies(user_id=1, n_recommendations=10)
```

Returns top 10 personalized movie recommendations with 
predicted ratings clipped between 0.5 and 5.0.

## 🔮 Future Improvements

- Implement content-based filtering using movie genres
- Add more training epochs with early stopping
- Deploy as a REST API using FastAPI and Docker
- Experiment with matrix factorization techniques

## 👤 Author

**Damian Ocampo Salles-Berges**  
Machine Learning Engineer  
[LinkedIn](https://linkedin.com/in/damiansalles) | 
[GitHub](https://github.com/dosb94)
