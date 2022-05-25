from src.features import build_features_consumer_dispute as build_consumer
from src.models import train_model_consumer_dispute as model_consumer

build_consumer.run()
model_consumer.run()
