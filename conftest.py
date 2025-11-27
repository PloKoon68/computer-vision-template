import sys
import os

# Proje kök dizinini bul
project_root = os.path.dirname(os.path.abspath(__file__))

# 'src' klasörünü Python path'ine ekle
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)