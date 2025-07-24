import os
import re
import numpy as np

def output_files():
    # Pulling the output files created from each model
    outputs = {}
    
    output_files = {'Naive': 'data/output/naive_output.txt', 'Traditional': 'data/output/traditional_output.txt', 'Deep Learning': 'data/output/deep_learning_output.txt'}
    
    for model_name, output_path in output_files.items():
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
                    outputs[model_name] = f.read()
            except Exception as e:
                print(f"Error reading {output_path}: {e}")
                outputs[model_name] = ""
        else:
            print(f"{output_path} not found")
            outputs[model_name] = ""
    
    return outputs

def parsing_outputs(output_text):
    # Parsing output from model output files
    final_recommendations = []
    
    recommendation_sections = output_text.strip().split('\n\n')
    
    for section in recommendation_sections:
        if section.strip():
            lines = section.strip().split('\n')
            if len(lines)>=2:
                title_line = lines[0]
                description_line = lines[1]
                
                matching_workout = re.search(r'Workout: (.+)', title_line)
                if matching_workout:
                    workout = matching_workout.group(1).strip()
                    
                    description_match = re.search(r'Description: (.+)', description_line)
                    if description_match:
                        description = description_match.group(1).strip()
                        final_recommendations.append({'title': workout, 'description': description})
    
    return final_recommendations

def quality_score_calculation(final_recommendations):
    # Calculating content quality based on descriptions
    quality_scores = []
    for recommendation in final_recommendations:
        description = recommendation['description']
        
        length_score = min(len(description)/100, 1.0)
        detail_score = description.count(',')/5
        instruction_score = 0
        instruction_words = ['how', 'step', 'position', 'form']
        for word in instruction_words:
            if word in description.lower():
                instruction_score = 1
                break
        
        total_score = (length_score + detail_score + instruction_score)/3
        quality_scores.append(total_score)
    
    final_quality_score = np.mean(quality_scores)
    
    return final_quality_score

def diversity_score_calculation(final_recommendations):
    # Calculating diversity score based on selected features
    all_features = []
    for recommendation in final_recommendations:
        description_lower = recommendation['description'].lower()
        
        feature_vector = {
            'strength': 1 if any(word in description_lower for word in ['strength', 'muscle', 'weight']) else 0,
            'cardio': 1 if any(word in description_lower for word in ['cardio', 'aerobic', 'endurance']) else 0,
            'flexibility': 1 if any(word in description_lower for word in ['stretch', 'flexibility']) else 0,
            'core': 1 if any(word in description_lower for word in ['core', 'stability']) else 0,
            'upper_body': 1 if any(word in description_lower for word in ['arm', 'chest', 'shoulder', 'back']) else 0,
            'lower_body': 1 if any(word in description_lower for word in ['leg', 'quad', 'hamstring']) else 0,
            'full_body': 1 if any(word in description_lower for word in ['full body', 'entire body']) else 0
        }
        all_features.append(feature_vector)
    
    unique_features = set()
    for feature in all_features:
        feature_tuple = tuple(feature.items())
        unique_features.add(feature_tuple)
    
    diversity_score = len(unique_features)/len(all_features)
    return diversity_score

def user_preference_words(final_recommendations, target_user_profile):
    # Creating user preference based on the set target profile
    user_preference_scores = []
    for recommendation in final_recommendations:
        description_lower = recommendation['description'].lower()
        score = 0
        
        # Experience level 
        experience = target_user_profile.get('experience_level', 'intermediate')
        if experience == 'beginner':
            beginner_words = ['simple', 'easy', 'basic', 'beginner']
            for word in beginner_words:
                if word in description_lower:
                    score += 0.4
                    break
        elif experience == 'advanced':
            advanced_words = ['advanced', 'challenging', 'complex', 'intense']
            for word in advanced_words:
                if word in description_lower:
                    score += 0.4
                    break
        
        # BMI
        bmi = target_user_profile.get('bmi', 25)
        if bmi < 18.5:  
            underweight_words = ['strength', 'muscle', 'weight']
            for word in underweight_words:
                if word in description_lower:
                    score += 0.3
                    break
        elif bmi > 30:
            obese_words = ['cardio', 'endurance', 'low impact']
            for word in obese_words:
                if word in description_lower:
                    score += 0.3
                    break
        
        user_preference_scores.append(score)
        user_preference_scores_mean = np.mean(user_preference_scores)
    
    return user_preference_scores_mean

def final_report():
    # Creates the final report with all the recommendations
    outputs = output_files()
    
    comparison = {}
    
    for model_name, output_text in outputs.items():
        final_recommendations = parsing_outputs(output_text)
        comparison[model_name] = {'recommendations': final_recommendations}
    
    # User profile for user word preference
    target_user = {
        'bmi': 22.5,
        'experience_level': 'intermediate'
    }
    
    evaluation_report = []
    
    # Calculate scores for all models
    model_scores = {}
    for model_name, data in comparison.items():
        final_recommendations = data['recommendations']
        model_scores[model_name] = {
            'diversity': diversity_score_calculation(final_recommendations),
            'quality': quality_score_calculation(final_recommendations),
            'preference': user_preference_words(final_recommendations, target_user)
        }
    
    for model_name, scores in model_scores.items():
        evaluation_report.append(f"{model_name} Model Scores:")
        evaluation_report.append(f"Diversity Score: {scores['diversity']:.3f}")
        evaluation_report.append(f"Content Quality Score: {scores['quality']:.3f}")
        evaluation_report.append(f"User Preference Score: {scores['preference']:.3f}")
        evaluation_report.append("")
    
    # Total score across all types
    evaluation_report.append("Overall Scores:")
    total_scores = {}
    for model, scores in model_scores.items():
        total_scores[model] = np.mean(list(scores.values()))
    
    sorted_total_scores = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (model, score) in enumerate(sorted_total_scores, 1):
        model_name = model.capitalize()
        formatted_score = f"{score:.3f}"
        evaluation_report.append(f"{model_name}: {formatted_score}")
    
    return "\n".join(evaluation_report)

def writing_to_file():
    # Saving the scores to the respective output file
    evaluation_report = final_report()
    
    output_path = 'data/output/evaluation_report.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(evaluation_report)
    
    return evaluation_report

if __name__ == "__main__":
    evaluation_report = writing_to_file()
    print("\n" + evaluation_report) 