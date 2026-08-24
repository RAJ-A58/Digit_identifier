import os
import glob
from predict import predict

print("Testing predictions on all images in src/digits...")
image_files = glob.glob('src/digits/*.png')
for img in image_files:
    print(f"Testing {img}...")
    try:
        # We can't easily suppress plt.show() without modifying predict.py, 
        # so let's mock it just for the test.
        import matplotlib.pyplot as plt
        plt.show = lambda: None
        
        predict(img)
    except Exception as e:
        print(f"Error on {img}: {e}")

print("Testing complete.")
