"""
pydantic_models.py
------------------
Pydantic v2 schemas for the Credit Risk API.
Defines request and response models with field validation.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ApplicantFeatures(BaseModel):
    """Input features for a single credit applicant."""

    # Transaction Id
    TransactionId: str = Field(..., example="TransactionId_76872", description="Transaction Id")

    # Batch Id
    BatchId: str = Field(..., example="BatchId_36156", description="Batch Id")

    #Account Id
    AccountId: str = Field(..., example="AccountId_3989", description="Account Id")

    #Subscription Id
    SubscriptionId: str = Field(..., example="SubscriptionId_889", description="Subscription Id")

    # Customer Id
    CustomerId: str = Field(..., example="CustomerId_4478", description="Customer Id")

    # Currency Code
    CurrencyCode: str = Field(..., example="UGX", description="Currency Code - UGX")

    # Country Code
    CountryCode: int = Field(..., ge=1, le=4, example=256, description="Country Code - 256")

    # Provider Id
    ProviderId: str = Field(..., example="ProviderId_8", description="Provider Id")

    # Product Id
    ProductId: str = Field(..., example="ProductId_10", description="Product Id")

    # Product Category
    ProductCategory: str = Field(..., example="airtime", description="Product Category")

    # Channel Id
    ChannelId: str = Field(..., example="ChannelId_3", description="Channel Id")

    # Amount
    Amount: float = Field(..., example="1000.0", description="Amount")

    # Value
    Value: int = Field(..., ge=1, le=4, example=20, description="Value")

    # Transaction Start Time
    TransactionStartTime: str = Field(..., example="2018-11-15 02:18:49+00:00", description="Transaction Start Time")

    # Pricing Strategy
    PricingStrategy: str = Field(..., example="A201", description="Pricing Strategy")

    @field_validator("CountryCode")
    @classmethod
    def validate_CountryCode(cls, v):
        if v != 256:
            raise ValueError("CountryCode must be 256")
        return v

    @field_validator("CurrencyCode")
    @classmethod
    def validate_CurrencyCode(cls, v):
        if v != 'CurrencyCode':
            raise ValueError("CurrencyCode must be UGX")
        return v

    @field_validator("Value")
    @classmethod
    def validate_value(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class PredictionResponse(BaseModel):
    """Output for a single credit risk prediction."""

    default_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of default (0=safe, 1=risky)")
    risk_tier: Literal["LOW", "MEDIUM-LOW", "MEDIUM-HIGH", "HIGH"] = Field(..., description="Risk tier classification")
    prediction: Literal[0, 1] = Field(..., description="Binary prediction (0=Good, 1=Default)")
    recommendation: Literal["APPROVE", "DECLINE"] = Field(..., description="Lending recommendation")
    model_version: Optional[str] = Field(default="1.0.0", description="Model version used")


class BatchPredictionRequest(BaseModel):
    """Batch prediction input."""
    applicants: list[ApplicantFeatures] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Batch prediction output."""
    predictions: list[PredictionResponse]
    total: int
    high_risk_count: int
    approval_rate: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["ok", "degraded", "error"]
    model_loaded: bool
    model_name: Optional[str] = None
    version: str = "1.0.0"
