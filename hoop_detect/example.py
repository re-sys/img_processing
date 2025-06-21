from circle_detector import CircleDetector
import cv2

def main():
    # Create detector instance with custom parameters
    detector = CircleDetector(
   # Minimum score threshold (0-1)
    )
    
    try:
        # Load and process image
        # Time the image loading and processing
        import time
        start_time = time.time()
        # cv2.imread("basket.jpg")
        detector.load_image("basket.jpg")  # Use your image path here
        detector.preprocess_image()
        
        # Detect circle using ROI-based approach
        result = detector.detect_circle()

        
        elapsed_time = time.time() - start_time
        print(f"Processing took {elapsed_time:.2f} seconds")
        
        if result:
            center, radius = result
            print(f"Circle detected!")
            print(f"Center: ({center[0]}, {center[1]})")
            print(f"Radius: {radius}")
            
            # Save result (optionally showing all ROIs)
            detector.save_result("result.jpg")
            print("Result saved as 'result.jpg'")
        else:
            print("No circle detected in the image")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 