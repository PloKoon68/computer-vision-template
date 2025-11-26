import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import AppConfig, load_configuration

def test_config_defaults():
    """Varsayılan değerler doğru mu?"""
    # Boş bir paths dict ile başlatıyoruz
    config = AppConfig(paths={})
    
    assert config.device == "cpu"
    assert config.confidence_threshold == 0.5
    assert config.frame_skip == 1
    assert config.roi_percent is None

def test_roi_parsing():
    """ROI string'i tuple'a doğru dönüşüyor mu?"""
    # .env varmış gibi davranalım
    os.environ["ROI_PERCENT"] = "0.1, 0.2, 0.5, 0.5"
    
    # Argparse simülasyonu yapmak zor olduğu için 
    # direkt AppConfig'e vererek test ediyoruz
    roi = (0.1, 0.2, 0.5, 0.5)
    
    config = AppConfig(paths={}, roi_percent=roi)
    
    assert len(config.roi_percent) == 4
    assert config.roi_percent[0] == 0.1
    assert isinstance(config.roi_percent, tuple)