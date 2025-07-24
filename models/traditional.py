import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    # Loading dataset preprocessed files
    exercises_preprocessed = pd.read_csv('data/preprocessed_data/exercises_preprocessed.csv')
    gym_members_preprocessed = pd.read_csv('data/preprocessed_data/gym_members_preprocessed.csv')

    return exercises_preprocessed, gym_members_preprocessed

def create_embeddings(exercises_preprocessed):
    # Prepare exercises for embedding
    exercises = []
    valid_exercises = []
   
    for idx, exercise in exercises_preprocessed.iterrows():
        # Picking out features
        feature_columns = ['Title', 'Desc', 'Type', 'BodyPart', 'Equipment', 'Level']
        features = []
       
        for column in feature_columns:
            if pd.notna(exercise[column]):
                feature_value = exercise[column]
            else:
                feature_value = ''
            features.append(feature_value)
       
        exercise_text = ' '.join(features)
        exercises.append(exercise_text)
        valid_exercises.append(exercise)
   
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    embeddings = tfidf.fit_transform(exercises)
   
    # Create mapping from exercise title to embedding
    exercise_embeddings = {}
    for i, exercise in enumerate(valid_exercises):
        exercise_embeddings[exercise['Title']] = embeddings[i].toarray().flatten()
   
    return exercise_embeddings

def similar_exercises(target_exercise, embeddings, top_k=5):
    # Use embeddings to find similar exercises
    target_embedding = embeddings[target_exercise]
    similarities = {}
   
    for exercise, embedding in embeddings.items():
        if exercise != target_exercise:
            similarity = cosine_similarity([target_embedding], [embedding])[0][0]
            similarities[exercise] = similarity
   
    # Sort by similarity and return 5 exercise values
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_similar_exercises = sorted_similarities[:top_k]
    
    return top_similar_exercises

def users_similarities(user_profile, gym_members_preprocessed):
    # Finding similar users and map workout to exercise type
    if user_profile['bmi'] < 18.5:
        bmi_category = 'underweight'
    elif user_profile['bmi'] < 25:
        bmi_category = 'normal'
    elif user_profile['bmi'] < 30:
        bmi_category = 'overweight'
    else:
        bmi_category = 'obese'
    
    similar_users = gym_members_preprocessed[
        (gym_members_preprocessed['Gender'].str.lower() == user_profile['gender'].lower()) |
        (gym_members_preprocessed['Experience_Level_Category'] == user_profile['experience_level'].lower()) |
        (gym_members_preprocessed['BMI_Category'] == bmi_category) |
        ((gym_members_preprocessed['Age'] >= user_profile['age'] - 5) & (gym_members_preprocessed['Age'] <= user_profile['age'] + 5))
    ]
    
    # Determine workout type from similar users
    if not similar_users.empty:
        workout_type = similar_users['Workout_Type'].value_counts().idxmax()
    else:
        workout_type = 'strength'
    
    # Mapping workout type to exercise type
    workout_to_exercise_mapping = {'strength': 'strength', 'cardio': 'cardio', 'hiit': 'cardio', 'yoga': 'strength'}
    exercise_type = workout_to_exercise_mapping.get(workout_type.lower(), 'strength')

    return exercise_type

def tfidf_recommendations(user_profile):
    exercises_preprocessed, gym_members_preprocessed = load_data()
   
    embeddings = create_embeddings(exercises_preprocessed)
    user_similarity = users_similarities(user_profile, gym_members_preprocessed)
   
    target_exercises = exercises_preprocessed[(exercises_preprocessed['Type'].str.lower() == user_similarity) & (exercises_preprocessed['Rating'].notna()) & (exercises_preprocessed['Rating'] > 0)].sort_values('Rating', ascending=False).head(3)
   
    recommendations = []
    used_exercises = set()
   
    for _, target_exercise in target_exercises.iterrows():
        if len(recommendations) >= 5:
            break
           
        top_similar_exercises = similar_exercises(target_exercise['Title'], embeddings, top_k=5)
       
        for exercise_title, similarity in top_similar_exercises:
            if len(recommendations) >= 5:
                break
               
            if exercise_title not in used_exercises:
                exercise_info = exercises_preprocessed[exercises_preprocessed['Title'] == exercise_title]
                if not exercise_info.empty:
                    exercise = exercise_info.iloc[0]
                    if pd.notna(exercise['Desc']) and exercise['Desc'].strip() != '':
                        recommendations.append({'title': exercise['Title'], 'description': exercise['Desc'], 'rating': exercise['Rating'], 'type': exercise['Type'], 'similarity': similarity})
                        used_exercises.add(exercise_title)
   
    return recommendations

def main():
    user_profile = {
        'gender': 'male',
        'age': 25,
        'bmi': 22.5,
        'experience_level': 'intermediate'
    }
   
    recommendations = tfidf_recommendations(user_profile)
    
    # Printing to output file
    with open('data/output/traditional_output.txt', 'w') as f:
        for recommendation in recommendations:
            f.write(f"Workout: {recommendation['title']}\n")
            f.write(f"Description: {recommendation['description'][:300]}{'...' if len(recommendation['description'])>300 else ''}\n")
            f.write("\n")

if __name__ == "__main__":
    main()