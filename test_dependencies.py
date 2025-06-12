#!/usr/bin/env python3
"""
Test script to verify all dependencies work correctly
"""

def test_imports():
    """Test that all required packages can be imported"""
    try:
        print("Testing imports...")
        
        import psycopg
        print("✅ psycopg imported successfully")
        
        import numpy as np
        print("✅ numpy imported successfully")
        
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
        
        import seaborn as sns
        print("✅ seaborn imported successfully")
        
        import scipy
        print("✅ scipy imported successfully")
        
        # Test basic functionality
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        print("✅ pandas DataFrame creation works")
        
        arr = np.array([1, 2, 3])
        print("✅ numpy array creation works")
        
        print("\n🎉 All dependencies working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing dependencies: {e}")
        return False

if __name__ == '__main__':
    success = test_imports()
    exit(0 if success else 1)