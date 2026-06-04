CATEGORICAL_COLS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "TransactionStartTime",
]

NUMERICAL_COLS = ["Amount", "Value," "PricingStrategy", "FraudResult"]

TARGET_COL = "FraudResult"


COLUMN_NAMES = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "Amount",
    "Value",
    "TransactionStartTime",
    "PricingStrategy",
    "FraudResult",  # 1 = Good, 2 = Bad
]
