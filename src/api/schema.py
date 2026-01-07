from pydantic import BaseModel


class ChurnRequest(BaseModel):
    age: int
    tenure: int
    usage_frequency: int
    support_calls: int
    payment_delay: int
    total_spend: float
    last_interaction: int
    gender: str
    contract_length: str
    subscription_type: str


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
