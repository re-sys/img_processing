import numpy as np
import cv2

class CircleDetector:
    def __init__(self, scale_factor=0.6):
        """
        Initialize the CircleDetector class.
        
        Args:
            scale_factor (float): Scale factor for image resizing (0-1). Default is 0.5 (half size).
        """
        self.image = None
        self.processed_image = None
        self.result = None
        self.scale_factor = scale_factor
        self.original_image = None
        self.original_size = None

    def load_image(self, image_path):
        """Load and store the image."""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Failed to load image from {image_path}")
            
        self.original_size = self.original_image.shape[:2]  # (height, width)
        
        # Resize image for processing
        height = int(self.original_size[0] * self.scale_factor)
        width = int(self.original_size[1] * self.scale_factor)
        self.image = cv2.resize(self.original_image, (width, height))
        
        print(f'Original shape: {self.original_size}, Resized shape: {self.image.shape}')
        return self

    def preprocess_image(self):
        """Preprocess the image using HSV color space and create mask."""
        if self.image is None:
            raise ValueError("No image loaded")

        # doing local histogram average

        # Convert to HSV and enhance saturation
        hsv_frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_frame)
        s = np.clip(s * 1.5, 0, 255).astype(np.uint8)
        hsv_enhanced = cv2.merge([h, s, v])
        rgb_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)


        # Create color mask
        lower_bound = np.array([5, 220, 0])
        upper_bound = np.array([11, 255, 255])
        mask = cv2.inRange(hsv_enhanced, lower_bound, upper_bound)
        mask = cv2.medianBlur(mask, 23)
        original_mask = mask.copy()
        # Clean up mask
        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # erode
        # mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=4)
        mask = cv2.medianBlur(mask, 23)
        mask = cv2.medianBlur(mask, 23)
        #canny 
        # setting the largest contour's bounding box into zero
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            original_mask[y:y+h, x:x+w] = 0
        # dilate
        # mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        # median blur
        # mask = cv2.medianBlur(mask, 23)
        cv2.imshow('original_mask', original_mask)
        cv2.waitKey(0)


        self.processed_image = mask
        return self

    def _build_matrices(self, points):
        """Build matrices for circle fitting."""
        points = np.array(points, dtype=np.float64)
        x = points[:, 0]
        y = points[:, 1]
        
        n = len(points)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_x2 = np.sum(x**2)
        sum_y2 = np.sum(y**2)
        sum_xy = np.sum(x * y)
        
        A = np.array([
            [sum_x2, sum_xy, sum_x],
            [sum_xy, sum_y2, sum_y],
            [sum_x,  sum_y,  n]
        ])
        
        x2y2 = x**2 + y**2
        B = np.array([
            np.sum(x * x2y2),
            np.sum(y * x2y2),
            np.sum(x2y2)
        ])
        
        return A, B

    def _fit_circle(self, points):
        """
        Fit circle to given points.
        
        Returns:
            tuple: ((center_x, center_y), radius) or (None, None)
        """
        if len(points) < 3:
            return None, None

        A, B = self._build_matrices(points)
        try:
            X = np.linalg.solve(A, B)
            u, v, w = X
            center_x = u / 2.0
            center_y = v / 2.0
            radius = np.sqrt(center_x**2 + center_y**2 + w)
            return (int(center_x), int(center_y)), int(radius)
        except np.linalg.LinAlgError:
            return None, None

    def _scale_to_original(self, center, radius):
        """
        Scale the detected circle parameters back to original image size.
        
        Args:
            center (tuple): (x, y) coordinates of circle center in resized image
            radius (int): radius in resized image
            
        Returns:
            tuple: ((original_x, original_y), original_radius)
        """
        if center is None or radius is None:
            return None, None
            
        original_x = int(center[0] / self.scale_factor)
        original_y = int(center[1] / self.scale_factor)
        original_radius = int(radius / self.scale_factor)
        
        return (original_x, original_y), original_radius

    def detect_circle(self):
        """
        Detect circle by fitting all points in the binary image.
        
        Returns:
            tuple: ((center_x, center_y), radius) or None
        """
        if self.processed_image is None:
            raise ValueError("Image not preprocessed")

        # Get points from binary image
        points = np.column_stack(np.where(self.processed_image > 0))
        if points.size == 0:
            return None

        # Convert (y,x) to (x,y)
        points = points[:, [1, 0]]
        
        # Fit circle using all points
        center, radius = self._fit_circle(points)
        
        if center is not None and radius is not None:
            # Scale back to original image size
            original_center, original_radius = self._scale_to_original(center, radius)
            self.result = (original_center, original_radius)
            return original_center, original_radius
        return None

    def draw_result(self):
        """
        Draw the detected circle on the original image.
        
        Returns:
            numpy.ndarray: Result image
        """
        if self.original_image is None:
            raise ValueError("No image loaded")

        result_image = self.original_image.copy()
        
        if self.result is not None:
            center, radius = self.result
            cv2.circle(result_image, center, radius, (0, 255, 0), 2)
            cv2.circle(result_image, center, 5, (0, 0, 255), -1)
        
        cv2.imshow("result_image", result_image)
        cv2.waitKey(0)
        return result_image

    def save_result(self, output_path):
        """
        Save the result image with detected circle.
        
        Args:
            output_path (str): Path to save the result image
        """
        result_image = self.draw_result()
        cv2.imwrite(output_path, result_image)
        return self 
if __name__ == "__main__":
    detector = CircleDetector()
    detector.load_image("basket.jpg")
    detector.preprocess_image()
    detector.detect_circle()
    detector.draw_result()
    # detector.save_result("result.jpg")