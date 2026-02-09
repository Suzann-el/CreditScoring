from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel
import uvicorn
from lightgbm import LGBMClassifier

# 
# --- Loading model ---
#

Credit_clf_final = pickle.load(open("model.pkl", 'rb'))

# 
# --- Initializing FastAPI ---
#


app = FastAPI()

# 
# --- Creating data Class (input) ---
#


class Client_data(BaseModel):
    FLAG_OWN_CAR: int
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: float
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: int
    OWN_CAR_AGE: float
    REGION_RATING_CLIENT_W_CITY: int
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DAYS_LAST_PHONE_CHANGE: float
    NAME_CONTRACT_TYPE_Cashloans: int
    NAME_EDUCATION_TYPE_Highereducation: int
    NAME_FAMILY_STATUS_Married: int
    DAYS_EMPLOYED_PERC: float
    INCOME_CREDIT_PERC: float
    ANNUITY_INCOME_PERC: float
    PAYMENT_RATE: float
    BURO_DAYS_CREDIT_MAX: float
    BURO_DAYS_CREDIT_MEAN: float
    BURO_DAYS_CREDIT_ENDDATE_MAX: float
    BURO_AMT_CREDIT_MAX_OVERDUE_MEAN: float
    BURO_AMT_CREDIT_SUM_MEAN: float
    BURO_AMT_CREDIT_SUM_DEBT_MEAN: float
    BURO_CREDIT_TYPE_Microloan_MEAN: float
    BURO_CREDIT_TYPE_Mortgage_MEAN: float
    ACTIVE_DAYS_CREDIT_MAX: float
    ACTIVE_DAYS_CREDIT_ENDDATE_MIN: float
    ACTIVE_DAYS_CREDIT_ENDDATE_MEAN: float
    ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN: float
    ACTIVE_AMT_CREDIT_SUM_SUM: float
    ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN: float
    CLOSED_DAYS_CREDIT_VAR: float
    CLOSED_AMT_CREDIT_SUM_MAX: float
    CLOSED_AMT_CREDIT_SUM_SUM: float
    PREV_APP_CREDIT_PERC_MIN: float
    PREV_APP_CREDIT_PERC_MEAN: float
    PREV_CNT_PAYMENT_MEAN: float
    PREV_NAME_CONTRACT_STATUS_Refused_MEAN: float
    PREV_NAME_YIELD_GROUP_low_action_MEAN: float
    PREV_PRODUCT_COMBINATION_CashXSelllow_MEAN: float
    APPROVED_AMT_ANNUITY_MEAN: float
    APPROVED_AMT_DOWN_PAYMENT_MAX: float
    APPROVED_CNT_PAYMENT_MEAN: float
    POS_MONTHS_BALANCE_MAX: float
    POS_MONTHS_BALANCE_SIZE: float
    POS_SK_DPD_DEF_MEAN: float
    INSTAL_DPD_MEAN: float
    INSTAL_DBD_SUM: float
    INSTAL_PAYMENT_PERC_SUM: float
    INSTAL_PAYMENT_DIFF_MEAN: float
    INSTAL_AMT_INSTALMENT_MAX: float
    INSTAL_AMT_PAYMENT_MIN: float
    INSTAL_AMT_PAYMENT_SUM: float
    INSTAL_DAYS_ENTRY_PAYMENT_MAX: float
    INSTAL_DAYS_ENTRY_PAYMENT_MEAN: float
    CC_CNT_DRAWINGS_CURRENT_MEAN: float
    CC_CNT_DRAWINGS_CURRENT_VAR: float

#
# --- Creating prediction Class (output) ---
#
# 

class Client_Target(BaseModel):
    prediction: int
    probability: float


#
# --- Welcome message ---
#

@app.get("/")
def read_root():
    return {"message": "Welcome to the API for Loan Default prediction"}


#
# --- POST request to get the predicted target + probability ---
#



@app.post('/predict', response_model=Client_Target)
def model_predict(input: Client_data):
    """Predict with input"""
    X = pd.json_normalize(input.__dict__)
    probability = float(Credit_clf_final.predict_proba(X)[:, 1])
    prediction = int((probability >= 0.4))
    return {
        'prediction': prediction,
        'probability': probability
    }


