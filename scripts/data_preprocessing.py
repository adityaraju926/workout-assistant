import pandas as pd

def preprocess_exercise_raw():
    exercises_raw = pd.read_csv('data/raw_data/exercises.csv')
    
    exercises_processed = exercises_raw.copy()

    # Cleaning the columns
    if 'level' in exercises_processed.columns:
        exercises_processed['level'] = exercises_processed['level'].str.lower()
    if 'category' in exercises_processed.columns:
        exercises_processed['category'] = exercises_processed['category'].str.lower()
    if 'equipment' in exercises_processed.columns:
        exercises_processed['equipment'] = exercises_processed['equipment'].fillna('body only')
    if 'primaryMuscles' in exercises_processed.columns:
        exercises_processed['primaryMuscles'] = exercises_processed['primaryMuscles'].str.lower()
    if 'secondaryMuscles' in exercises_processed.columns:
        exercises_processed['secondaryMuscles'] = exercises_processed['secondaryMuscles'].str.lower()
    if 'force' in exercises_processed.columns:
        exercises_processed['force'] = exercises_processed['force'].str.lower()
    if 'mechanic' in exercises_processed.columns:
        exercises_processed['mechanic'] = exercises_processed['mechanic'].str.lower()
    
    return exercises_processed

def preprocess_gym_members_raw():
    gym_members_raw=pd.read_csv('data/raw_data/gym_members.csv')
    
    gym_members_processed=gym_members_raw.copy()
    
    # Cleaning the columns
    if 'Gender' in gym_members_processed.columns:
        gym_members_processed['Gender']=gym_members_processed['Gender'].str.lower()
    if 'Workout_Type' in gym_members_processed.columns:
        gym_members_processed['Workout_Type']=gym_members_processed['Workout_Type'].str.lower()
    
    # Categorizing the experience levels to match the UI
    if 'Experience_Level' in gym_members_processed.columns:
        def experience_categories(level):
            if pd.isna(level):
                return 'unknown'
            elif level == 1:
                return 'beginner'
            elif level == 2:
                return 'intermediate'
            elif level == 3:
                return 'advanced'
            else:
                return 'unknown'
        
        gym_members_processed['Experience_Level_Category']=gym_members_processed['Experience_Level'].apply(experience_categories)
    
    # Categorizing the BMI values to match the data
    if 'BMI' in gym_members_processed.columns:
        def bmi_categories(bmi):
            if pd.isna(bmi):
                return 'unknown'
            elif bmi < 18.5:
                return 'underweight'
            elif bmi < 25:
                return 'normal'
            elif bmi < 30:
                return 'overweight'
            else:
                return 'obese'
        
        gym_members_processed['BMI_Category']=gym_members_processed['BMI'].apply(bmi_categories)
    
    return gym_members_processed

def save_processed_data():
    preprocessed_exercise = preprocess_exercise_raw()
    preprocessed_exercise.to_csv('data/preprocessed_data/exercises_preprocessed.csv', index=False)
    
    preprocessed_gym_members = preprocess_gym_members_raw()
    preprocessed_gym_members.to_csv('data/preprocessed_data/gym_members_preprocessed.csv', index=False)

if __name__ == "__main__":
    save_processed_data() 