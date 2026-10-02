"""ergonomy module"""

from .components import Screen
from .models import User

__all__ = ["calculate_incidence_angle"]

def calculate_incidence_angle(user:User, screen:Screen):
    NotImplementedError