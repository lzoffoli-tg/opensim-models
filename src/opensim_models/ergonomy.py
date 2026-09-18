"""ergonomy module"""

from .models import User, Screen

def __all__ = ["calculate_incidence_angle"]

def calculate_incidence_angle(user:User, screen:Screen):
    NotImplementedError