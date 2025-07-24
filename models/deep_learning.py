import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def calculate_bmi(height_inches, weight_lbs):
    # Calculating BMI
    height_meters=height_inches*0.025
    weight_kg=weight_lbs*0.453
    bmi=weight_kg/(height_meters**2)
    
    return bmi

def load_data():
    # Loading dataset preprocessed files
    exercises_preprocessed = pd.read_csv('data/preprocessed_data/exercises_preprocessed.csv')
    gym_members_preprocessed = pd.read_csv('data/preprocessed_data/gym_members_preprocessed.csv')
    
    return exercises_preprocessed, gym_members_preprocessed

def user_exercise(gym_members_preprocessed, exercises_preprocessed):
    mappings = []
    
    # Map workout types to exercise types
    workout_to_exercise_mapping = {
        'strength': 'strength',
        'cardio': 'cardio',
        'hiit': 'cardio',
        'yoga': 'stretching',  
        'pilates': 'stretching',  
        'crossfit': 'strength',  
        'bodybuilding': 'strength',  
        'powerlifting': 'powerlifting',
        'olympic': 'olympic weightlifting',
        'strongman': 'strongman',
        'plyometrics': 'plyometrics'
    }
   
    for _, user in gym_members_preprocessed.iterrows():
        user_workout_type = user['Workout_Type'].lower()
       
        # Mapping workout type to exercise type
        exercise_type = workout_to_exercise_mapping.get(user_workout_type, 'strength')
        matching_exercises = exercises_preprocessed[exercises_preprocessed['Type'].str.lower()==exercise_type]
       
        if not matching_exercises.empty:
            top_exercises = matching_exercises.sort_values('Rating', ascending=False).head(10)
           
            for _, exercise in top_exercises.iterrows():
                # Creating interaction score based on rating and user preferences
                interaction_score = exercise['Rating']/10.0
    
                interaction_score += np.random.normal(0, 0.1)
                interaction_score = np.clip(interaction_score, 0, 1)
               
                mappings.append({'user_id': user.name, 'exercise_id': exercise.name, 'rating': interaction_score, 'workout_type': user_workout_type, 'exercise_type': exercise['Type'].lower()})

        refined_maps = pd.DataFrame(mappings)
    return refined_maps

def encoding_features(gym_members_preprocessed, exercises_preprocessed, user_exercise_df):
    # Encoding categorical features
    gym_members_encoding = {}
    exercises_encoding = {}
   
    # User features
    gym_members_encoding['gender'] = LabelEncoder()
    gym_members_encoding['experience_level'] = LabelEncoder()
    gym_members_encoding['bmi_category'] = LabelEncoder()
    gym_members_encoding['workout_type'] = LabelEncoder()
   
    # Exercise features
    exercises_encoding['type'] = LabelEncoder()
    exercises_encoding['bodypart'] = LabelEncoder()
    exercises_encoding['equipment'] = LabelEncoder()
    exercises_encoding['level'] = LabelEncoder()
   
    user_features = {}
    # Ensure all categorical columns are strings and handle any missing values
    user_features['gender'] = gym_members_encoding['gender'].fit_transform(gym_members_preprocessed['Gender'].astype(str).fillna('unknown'))
    user_features['experience_level'] = gym_members_encoding['experience_level'].fit_transform(gym_members_preprocessed['Experience_Level_Category'].astype(str).fillna('unknown'))
    user_features['bmi_category'] = gym_members_encoding['bmi_category'].fit_transform(gym_members_preprocessed['BMI_Category'].astype(str).fillna('unknown'))
    user_features['workout_type'] = gym_members_encoding['workout_type'].fit_transform(gym_members_preprocessed['Workout_Type'].astype(str).fillna('unknown'))
   
    scaler = StandardScaler()
    user_features['age'] = scaler.fit_transform(gym_members_preprocessed[['Age']]).flatten()
    user_features['bmi'] = scaler.fit_transform(gym_members_preprocessed[['BMI']]).flatten()
   
    exercise_features = {}
    # Ensure all categorical columns are strings and handle any missing values
    exercise_features['type'] = exercises_encoding['type'].fit_transform(exercises_preprocessed['Type'].astype(str).fillna('unknown'))
    exercise_features['bodypart'] = exercises_encoding['bodypart'].fit_transform(exercises_preprocessed['BodyPart'].astype(str).fillna('unknown'))
    exercise_features['equipment'] = exercises_encoding['equipment'].fit_transform(exercises_preprocessed['Equipment'].astype(str).fillna('unknown'))
    exercise_features['level'] = exercises_encoding['level'].fit_transform(exercises_preprocessed['Level'].astype(str).fillna('unknown'))
   
    exercise_features['rating'] = exercises_preprocessed['Rating'].fillna(0)/10.0
   
    return user_features, exercise_features, gym_members_encoding, exercises_encoding, scaler

def neural_collaborative_model(user_number, exercise_number, embedding_dim=64):
    # Building a neural collaborative filtering model
    user_input = layers.Input(shape=(1,), name='user_input')
    user_embedding = layers.Embedding(user_number, embedding_dim, name='user_embedding')(user_input)
    user_embedding = layers.Flatten()(user_embedding)
   
    exercise_input = layers.Input(shape=(1,), name='exercise_input')
    exercise_embedding = layers.Embedding(exercise_number, embedding_dim, name='exercise_embedding')(exercise_input)
    exercise_embedding = layers.Flatten()(exercise_embedding)
   
    # Combining embeddings
    concat = layers.Concatenate()([user_embedding, exercise_embedding])
   
    dense1 = layers.Dense(128, activation='relu')(concat)
    dropout1 = layers.Dropout(0.3)(dense1)
    dense2 = layers.Dense(64, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.2)(dense2)
    dense3 = layers.Dense(32, activation='relu')(dropout2)
   
    # Output layer
    output_layer = layers.Dense(1, activation='sigmoid', name='rating')(dense3)
    model = keras.Model(inputs=[user_input, exercise_input], outputs=output_layer)
   
    return model

def training_model(user_exercise_df, user_features, exercise_features, epochs=10, batch_size=32):
    # Training the neural collaborative filtering model
    gym_members_ids = user_exercise_df['user_id'].values
    exercises_ids = user_exercise_df['exercise_id'].values
    ratings = user_exercise_df['rating'].values
   
    user_train, user_test, exercise_train, exercise_test, y_train, y_test = train_test_split(gym_members_ids, exercises_ids, ratings, test_size=0.2, random_state=42)
   
    user_number = len(user_features['gender'])
    exercise_number = len(exercise_features['type'])
   
    model = neural_collaborative_model(user_number, exercise_number)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
   
    # Train model
    history = model.fit([user_train, exercise_train], y_train, validation_data=([user_test, exercise_test], y_test), epochs=epochs, batch_size=batch_size, verbose=0)
   
    return model, history

def exercise_embeddings(model, exercise_id):
    # Extracting exercise embeddings
    exercise_embedding_layer = model.get_layer('exercise_embedding')
    exercise_embedding = exercise_embedding_layer.get_weights()[0][exercise_id]
   
    return exercise_embedding

def similar_exercises(model, target_exercise_id, exercises_preprocessed, top_k=10):
    # Finding similar exercises using embeddings
    target_embedding = exercise_embeddings(model, target_exercise_id)
   
    similarities = []
    for idx, exercise in exercises_preprocessed.iterrows():
        if idx != target_exercise_id:
            exercise_embedding = exercise_embeddings(model, idx)
            similarity = np.dot(target_embedding, exercise_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(exercise_embedding)
            )
            similarities.append((idx, similarity))
   
    similarities.sort(key=lambda x: x[1], reverse=True)
    final_similarities = similarities[:top_k]

    return final_similarities

def similar_users(user_profile, gym_members_preprocessed, gym_members_encoding, exercises_encoding):
    # Finding similar user using embeddings
    best_match = None
    best_score = -1
   
    for idx, user in gym_members_preprocessed.iterrows():
        score = 0
        # Gender similarity
        if str(user['Gender']).lower() == user_profile['gender'].lower():
            score += 2
       
        # Experience level similarity
        if str(user['Experience_Level_Category']).lower() == user_profile['experience_level']:
            score += 2
       
        # BMI categorizing and similarity
        if user_profile['bmi'] < 18.5:
            user_bmi_category = 'underweight'
        elif user_profile['bmi'] < 25:
            user_bmi_category = 'normal'
        elif user_profile['bmi'] < 30:
            user_bmi_category = 'overweight'
        else:
            user_bmi_category = 'obese'
        
        if str(user['BMI_Category']).lower() == user_bmi_category:
            score += 1.5
       
        if abs(float(user['Age']) - user_profile['age']) <= 5:
            score += 1
       
        if score > best_score:
            best_score = score
            best_match = user
   
    if best_match is None and len(gym_members_preprocessed) > 0:
        best_match = gym_members_preprocessed.iloc[0]
   
    return best_match

def recommending_exercises(user_profile, model, gym_members_preprocessed, exercises_preprocessed, user_features, exercise_features, gym_members_encoding, exercises_encoding, num_recommendations=5):
    # Generating recommendations using neural collaborative filtering
    similar_user = similar_users(user_profile, gym_members_preprocessed, gym_members_encoding, exercises_encoding)
    similar_user_workout = similar_user['Workout_Type'].lower()

    workout_to_exercise_mapping = {
        'strength': 'strength',
        'cardio': 'cardio',
        'hiit': 'cardio',
        'yoga': 'stretching',  
        'pilates': 'stretching',  
        'crossfit': 'strength',  
        'bodybuilding': 'strength',  
        'powerlifting': 'powerlifting',
        'olympic': 'olympic weightlifting',
        'strongman': 'strongman',
        'plyometrics': 'plyometrics'
    }
    
    exercise_type = workout_to_exercise_mapping.get(similar_user_workout, 'strength')
    
    target_exercises = exercises_preprocessed[exercises_preprocessed['Type'].str.lower() == exercise_type].sort_values('Rating', ascending=False).head(3)
   
    recommendations = []
    used_exercises = set()
   
    for _, target_exercise in target_exercises.iterrows():
        if len(recommendations) >= num_recommendations:
            break
       
        # Finding similar exercises
        similar_exercises_list = similar_exercises(
            model, target_exercise.name, exercises_preprocessed, top_k=15
        )
        
        for exercise_id, similarity in similar_exercises_list:
            if len(recommendations) >= num_recommendations:
                break
           
            exercise = exercises_preprocessed.iloc[exercise_id]
            if (exercise['Title'] not in used_exercises and
                pd.notna(exercise['Desc']) and
                exercise['Desc'].strip() != ''):
               
                recommendations.append({
                    'title': exercise['Title'],
                    'description': exercise['Desc'],
                    'similarity': similarity,
                })
                used_exercises.add(exercise['Title'])
   
    return recommendations

def final_recommendations(gender, age, height_inches, weight_lbs, experience_level):
    bmi = calculate_bmi(height_inches, weight_lbs)
    
    user_profile = {
        'gender': gender.lower(),
        'age': age,
        'bmi': bmi,
        'experience_level': experience_level.lower()
    }
    
    exercises_preprocessed, gym_members_preprocessed = load_data()
   
    user_exercise_df = user_exercise(gym_members_preprocessed, exercises_preprocessed)
    user_features, exercise_features, gym_members_encoding, exercises_encoding, scaler = encoding_features(gym_members_preprocessed, exercises_preprocessed, user_exercise_df)
   
    model, _ = training_model(user_exercise_df, user_features, exercise_features, epochs=10)
    recommendations = recommending_exercises(user_profile, model, gym_members_preprocessed, exercises_preprocessed, user_features, exercise_features, gym_members_encoding, exercises_encoding)
   
    return recommendations