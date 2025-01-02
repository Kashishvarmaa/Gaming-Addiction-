import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Define the data model for the request body
class UserInput(BaseModel):
    Age: int
    PlayTimeHours: float
    InGamePurchases: int
    GameDifficulty: int
    SessionsPerWeek: int
    AvgSessionDurationMinutes: float
    PlayerLevel: int
    AchievementsUnlocked: int
    EngagementLevel: int
    Gender: str
    Location: str
    GameGenre: str

# Load the pre-trained model and the label encoder
rf_model = joblib.load("rf_model.pkl")
le = joblib.load("label_encoder.pkl")
features = joblib.load("features.pkl")

@app.post("/predict")
def predict(user_input: UserInput):
    # Convert input data to the correct format
    user_data = {
        'Age': user_input.Age,
        'PlayTimeHours': user_input.PlayTimeHours,
        'InGamePurchases': user_input.InGamePurchases,
        'GameDifficulty': user_input.GameDifficulty,
        'SessionsPerWeek': user_input.SessionsPerWeek,
        'AvgSessionDurationMinutes': user_input.AvgSessionDurationMinutes,
        'PlayerLevel': user_input.PlayerLevel,
        'AchievementsUnlocked': user_input.AchievementsUnlocked,
        'EngagementLevel': user_input.EngagementLevel,
        'Gender_Male': 1 if user_input.Gender.lower() == 'male' else 0,
    }
    
    predefined_locations = ['US', 'UK', 'IN']
    for loc in predefined_locations:
        user_data[f'Location_{loc}'] = 1 if user_input.Location.upper() == loc else 0
    
    predefined_genres = ['Action', 'RPG', 'Puzzle']
    for genre in predefined_genres:
        user_data[f'GameGenre_{genre}'] = 1 if user_input.GameGenre.lower() == genre.lower() else 0
    
    # Convert the input data to a dataframe
    new_user_df = pd.DataFrame([user_data])
    
    # Ensure all features are present
    for col in features:
        if col not in new_user_df:
            new_user_df[col] = 0  # Add missing columns with default value 0
    
    # Reorder columns to match the training set
    new_user_df = new_user_df[features]
    
    # Make prediction
    new_user_prediction = rf_model.predict(new_user_df)
    addiction_level = le.inverse_transform(new_user_prediction)[0]
    
    # Provide tips based on predictions
    def provide_tips(addiction_level):
        if addiction_level == 'low':
            tips = "Your gaming habits seem balanced. Continue to enjoy gaming in moderation."
        elif addiction_level == 'mid':
            tips = "Consider monitoring your gaming habits and setting time limits to avoid potential addiction."
        elif addiction_level == 'high':
            tips = "It is important to take steps to reduce gaming time and seek help if needed. Consider engaging in other activities."
        return tips

    tips = provide_tips(addiction_level)
    
    return {"AddictionLevel": addiction_level, "Tips": tips}
