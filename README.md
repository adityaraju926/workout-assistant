# Workout Assistant

## Problem Statement

In the fitness industry whether you are a "beginner" or "advanced" individual at the gym, there are instances where you may need workouts that are personalized. However, many applications or blogs currently provide general information on the exercises that should be performed based on the area you want to improve, aligning with widely-known industry standards.

This project addresses these challenges by using a recommendation system that leverages machine learning to provide personalized workout suggestions tailored to individual user profiles.

## Data Source(s)

The project utilizes two primary datasets:

- **Gym Members Exercise Dataset** data/raw_data/gym_members.csv: This [dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset) contains information about gym members demographics and their workout types.
- **Gym Exercise Dataset** data/raw_data/exercises.csv: This [dataset](https://www.gigasheet.com/sample-data/gym-exercise-dataset) contains information strictly for workouts. Information such as type, body part, equipment, level, etc. are all found in this dataset.

## Review of Relevant Previous Efforts and Literature

### Existing Approaches

There are many attempts of applications that complete the same objective as this project by providing perseonalized exercises and/or diet plans based on user preferences and demographics. The following are examples:

- **[Fitness Recommender System](https://github.com/dilshankarunarathne/personalized-fitness-recommender-system)**
- **[TailoredFit - Personalized Home Workout Recommendations](https://github.com/RalphGradien/HomeWorkoutRecommendations)**
- **[Workout Recommender](https://github.com/itsLeonB/workout-recommender)**
- **[SmartFit - AI Powered Workout Website](https://github.com/manishtmtmt/ai-powered-workout-plan)** 

### Differences in this project
- **Model Comparison**: This project compares three different approaches, while majority of the other applications focus on using one approach.
- **Evaluation Strategy**: This project uses a greater variety of evaluation metrics such as diversity, content quality, user preference match, and performance ranking. The other projects use a single evaluation metric.
- **Research-Oriented**: This point is based on the prior two, but because of using more models and more complex evaluation metrics, this application is designed primarily for model comparison versus focusing on having a "production-ready" application.

## Metrics and Models Used

### Metrics

The evaluation process used in this project are based on the output files generated from each model. The following are the metrics used and each are on a 0-1 scale:

- **Diversity Score**: This metric measures the uniqueness of the workouts provided from each model by comparing it to all the different "types" of workouts. It works by creating a feature vector for each "type" of exercise (strength, cardio, etc.) and calculates the ratio against the total exercisese in the dataset. A higher score in this scenario means that the recommended exercises cover more "types."
 
- **Content Quality Score**: This metric measures how informative the descriptions are in the output file per model. It combines the description length, detail level, and presence of keywords which is then averaged for the final "Content Quality Score."

- **User Preference Score**: This metric measures how well the recommendations line up with the users demographics inputted. It compares the gender, experience level, and the rest of the inputted values  

- **Overall Score**
- Average of all evaluation metrics
- Provides overall model performance ranking

### Modeling Approach

- **Naive Approach (Rule-Based)**: This model uses simple attribute matching and filtering. It firsts loads the user and exercise data and then filters it based on the users characteristics. Then, it finds similar values in the dataset and determines what type of workout is preferred based on the rating provided for each. The top N recommendations that match the demographics of the user are then provided as recommendations.

- **Traditional Machine Learning (TF-IDF Collaborative Filtering)**: This model uses TF-IDF to create the exercise embeddings from the description, type, body parts, and difficulty. Then, cosine similarity is used to find similar exercises and users based on the prior embeddings to provide the exercise recommendations using content-based similarity matching.

- **Deep Learning (Neural Collaborative Filtering)**: This model uses a collaborative filtering approach by first building neural networks with embedding layers for users and exercises. Then, it trains on the patterns using binary cross-entropy loss and extracts the learned embeddings to find similar exercises and users. Then, the recommendations are provided based on the NN predictions.

## Data Preprocessing Steps

- First, the raw data file are loaded from the data/raw_data directory.
- Next, the data is cleaned and normalized. The steps here include converting text data to lowercase, handling missing values, and data validation
- Then, steps for feature engineering are done by converting the BMI into ranges, categorizing the levels into beginner, intermediate, and advanced.
- Finally, the columns are checked to ensure that the data is consistent and the outputs are written to their respective preprocessed file.

## Model Performance Comparison

| Metric | Naive Model | Traditional Model | Deep Learning Model |
|--------|-------------|------------------|-------------------|
| Diversity Score | 0.800 | 0.400 | 1.000 |
| Content Quality Score | 0.507 | 0.547 | 0.560 |
| User Preference Score | 0.180 | 0.060 | 0.180 |
| **Overall Score** | **0.496** | **0.336** | **0.580** |

The comparison shows that the Deep Learning model provides the most comprehensive recommendations across all metrics, with the Naive model offering a good balance of performance and interpretability.

## Results

1. **Deep Learning**: Best overall performance (0.580)
2. **Naive**: Good balanced approach (0.496)
3. **Traditional**: Limited performance (0.336)

## Ethics Statement

This project follows ethical practices and user safety. All user data is anonymized to protect individual privacy, with no PII stored or processed. The recommendation system is designed to work for individuals across all demographics, with the models accounting for various difficulty levels. The model decisions are transparent and limitations are clearly defined.

## Installing dependencies

```bash
pip install -r requirements.txt
```

### Project Structure:
```
workout-assistant/
├── data/
│   ├── raw_data/           
│   ├── preprocessed_data/  
│   └── output/            
├── scripts/
│   ├── data_preprocessing.py  
│   └── evaluation.py         
├── models/
│   ├── naive.py             
│   ├── traditional.py       
│   └── deep_learning.py     
├── UI.py                    
└── requirements.txt         
```

### Running the Application:

```bash
streamlit run UI.py
```

---