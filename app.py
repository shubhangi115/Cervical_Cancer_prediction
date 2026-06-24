import json
from fastapi import FastAPI,Path,HTTPException,Query
from pydantic import BaseModel,computed_field,Field
from typing import Annotated,Literal,Optional
from fastapi.responses import JSONResponse
import pandas as pd

# import the ml model
import joblib

model = joblib.load("cancer_model.joblib")


app= FastAPI()

# pydantic model to validate incoming data 
class UserInput(BaseModel):
    Age: Annotated[int, Field(..., gt=0, description="Age of the patient")]

    Number_of_sexual_partners: Annotated[int, Field(..., description="Number of sexual partners")]

    First_sexual_intercourse: Annotated[int, Field(..., description="Age at first sexual intercourse")]

    Num_of_pregnancies: Annotated[int, Field(..., description="Number of pregnancies")]

    Smokes: Annotated[int, Field(..., description="Smoker (0/1)")]

    Smokes_years: Annotated[float, Field(..., description="Years of smoking")]

    Smokes_packs_per_year: Annotated[float, Field(..., description="Packs per year")]

    Hormonal_Contraceptives: Annotated[int, Field(..., description="Hormonal contraceptives usage")]

    Hormonal_Contraceptives_years: Annotated[float, Field(..., description="Years of usage")]

    IUD: Annotated[int, Field(..., description="IUD usage")]

    IUD_years: Annotated[float, Field(..., description="Years of IUD usage")]

    STDs: Annotated[int, Field(..., description="History of STDs")]

    # It is basically a risk detector flag
    @computed_field
    @property
    def high_risk_flag(self) -> str:
        if self.Smokes and self.STDs and self.Number_of_sexual_partners > 3:
            return "High risk"
        return "Low risk"

    # if person is a heavy smoker or not
    @computed_field
    @property
    def smoking_intensity(self) -> str:
        intensity = self.Smokes_years * self.Smokes_packs_per_year

        return f"{intensity} Smoking score, the high it is the high risk is"

# endpoint for prediction
@app.post('/predict')
def predict_model(data: UserInput):

    # we will create a dataframe to give to the model, basically a row of all inputs, eg we have 12 inputs

    input_data = pd.DataFrame([{
        "Age": data.Age,
        "Number of sexual partners": data.Number_of_sexual_partners,
        "First sexual intercourse": data.First_sexual_intercourse,
        "Num of pregnancies": data.Num_of_pregnancies,

        "Smokes": data.Smokes,
        "Smokes (years)": data.Smokes_years,
        "Smokes (packs/year)": data.Smokes_packs_per_year,

        "Hormonal Contraceptives": data.Hormonal_Contraceptives,
        "Hormonal Contraceptives (years)": data.Hormonal_Contraceptives_years,

        "IUD": data.IUD,
        "IUD (years)": data.IUD_years,

        "STDs": data.STDs
    }])

    # Instead of manually building DataFrame:
    
    # input_df = pd.DataFrame([input_data])[expected_cols]
    # input_df = input_df.reindex(columns=model.feature_names_in_)
    
    # prediction=model.predict() # for prediction
    # return {
    #     "prediction": prediction.tolist()
    # }
    prediction=model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    if proba < 0.3:
        risk_level = "Low Risk"
    elif proba < 0.7:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return {
        "prediction": prediction,
        "risk_probability": round(float(proba), 4),
        "risk_percentage": round(float(proba * 100), 2),
        "risk_level": risk_level
    }


