import os
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, cfg):
        """Create model and store configurations"""
        self.model = None
        self.cfg = cfg
        self.save_dir = cfg["data"]["model_save_dir"]
        self.img_size = cfg["data"]["img_size"]
        self.num_classes = cfg["data"]["num_classes"]
        self.results_save_dir = cfg["results"]["save_dir"]
        
    @abstractmethod
    def train(self):
        """Method that runs full trainig process"""
        pass
    
    @abstractmethod
    def predict(self, **kwargs):
        """Method that returns predictions of model"""
        pass
    
    @abstractmethod
    def save_model(self):
        """Save model to local storage"""
        pass

    @abstractmethod
    def load_model(self, path):
        """Loads trained model from local weights"""
        pass

    