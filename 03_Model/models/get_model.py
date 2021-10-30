"""
"""
from models.FR_model import FRModel

def get_model(model_str: str):
    
    if model_str == 'FR_model':
        return FRModel
    else:
        raise NameError(f"There isn't {model_str}")

if __name__ == '__main__':
    pass

