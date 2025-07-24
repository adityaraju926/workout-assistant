import pandas as pd

def load_data():
    # Loading the preprocessed data from both datasets
    exercises_preprocessed = pd.read_csv('data/preprocessed_data/exercises_preprocessed.csv')
    gym_members_preprocessed = pd.read_csv('data/preprocessed_data/gym_members_preprocessed.csv')

    return exercises_preprocessed, gym_members_preprocessed

def processing_attributes(attribute, value, exercises_preprocessed, gym_members_preprocessed, used_workouts):
    if attribute == 'gender':
        cleaned_attributes = gym_members_preprocessed[gym_members_preprocessed['Gender'].str.lower()==value.lower()]
    elif attribute == 'experience_level':
        cleaned_attributes = gym_members_preprocessed[gym_members_preprocessed['Experience_Level_Category']==value.lower()]
    elif attribute == 'bmi':
        if value < 18.5:
            bmi_category = 'underweight'
        elif value < 25:
            bmi_category = 'normal'
        elif value < 30:
            bmi_category = 'overweight'
        else:
            bmi_category = 'obese'
        cleaned_attributes = gym_members_preprocessed[gym_members_preprocessed['BMI_Category']==bmi_category]
    elif attribute == 'age':
        cleaned_attributes = gym_members_preprocessed[(gym_members_preprocessed['Age']>=value-5) & (gym_members_preprocessed['Age']<=value+5)]
    else:
        return []
   
    workout_category = cleaned_attributes['Workout_Type'].value_counts().idxmax()
    exercise_category = {'strength': 'strength', 'cardio': 'cardio', 'hiit': 'cardio', 'yoga': 'strength'}.get(workout_category.lower(), 'strength')
    
    # Use column values that are rated and not empty
    rated_values = exercises_preprocessed[(exercises_preprocessed['Type'].str.lower()==exercise_category) & (exercises_preprocessed['Rating'].notna()) & (exercises_preprocessed['Rating']>0)]
    final_exercises = rated_values[rated_values['Title'].isin(used_workouts) == False].sort_values('Rating', ascending=False)
   
    all_workouts = []
    for _, exercise in final_exercises.iterrows():
        if pd.isna(exercise['Desc']) or exercise['Desc'].strip() == '':
            continue
           
        all_workouts.append({'attribute': attribute, 'attribute_value': value, 'workout_type': workout_category, 'title': exercise['Title'], 'description': exercise['Desc']})
   
    return all_workouts

def workout_recommendations(user_profile):
    exercises_preprocessed, gym_members_preprocessed = load_data()
    recommendations = []
    used_workouts = set()
   
    while len(recommendations) < 5:
        for attribute, value in user_profile.items():
            if len(recommendations) >= 5:
                break
            all_workouts = processing_attributes(attribute, value, exercises_preprocessed, gym_members_preprocessed, used_workouts)
            if all_workouts:
                recommendations.append(all_workouts[0])
                used_workouts.add(all_workouts[0]['title'])
   
    return recommendations

def main():
    user_profile = {
        'gender': 'male',
        'age': 25,
        'bmi': 22.5,
        'experience_level': 'intermediate'
    }
    recommendations = workout_recommendations(user_profile)
   
   # Printing to output file
    with open('data/output/naive_output.txt', 'w') as f:
        for recommendation in recommendations:
            f.write(f"Workout: {recommendation['title']}\n")
            f.write(f"Description: {recommendation['description'][:300]}{'...' if len(recommendation['description'])>300 else ''}\n")
            f.write("\n")

if __name__ == "__main__":
    main()