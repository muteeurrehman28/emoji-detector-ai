import io
import math
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image

def capture_and_stitch(output_path, base_name):
    # Fixed x-axis cropping coordinates (permanent values)
    start_x = 497
    end_x = 857

    # Initialize Chrome driver and set an initial window size
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.set_window_size(1920, 1080)

    # Open your target URL (include base_name query parameter)
    url = f"http://127.0.0.1:5000/?base_name={base_name}"
    driver.get(url)
    time.sleep(3)  # Wait for page to load fully

    # Get total page height and viewport height
    total_height = driver.execute_script("return document.body.scrollHeight")
    viewport_height = driver.execute_script("return window.innerHeight")
    print(f"Total page height: {total_height}px, Viewport height: {viewport_height}px")

    # Calculate number of screenshots needed
    num_screens = math.ceil(total_height / viewport_height)
    print(f"Capturing {num_screens} screenshots for full-page view.")

    # List to hold individual screenshots
    screenshots = []

    for i in range(num_screens):
        if i == num_screens - 1:
            scroll_y = total_height - viewport_height
        else:
            scroll_y = i * viewport_height
        driver.execute_script("window.scrollTo(0, arguments[0]);", scroll_y)
        time.sleep(1)  # Allow time for scroll
        png = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png))
        screenshots.append(img)

    driver.quit()

    if len(screenshots) < 2:
        print("Not enough screenshots captured to perform custom stitching.")
        return

    # Crop first screenshot (all vertical content starting from 20px from top)
    first_img = screenshots[0]
    first_cropped = first_img.crop((start_x, 20, end_x, first_img.height))

    # For the second screenshot, take only the bottom 20% of it
    second_img = screenshots[1]
    second_height = second_img.height - 20
    start_y_second_crop = int(second_height * 0.791)
    second_cropped = second_img.crop((start_x, start_y_second_crop, end_x, second_height))

    final_width = end_x - start_x
    final_height = first_cropped.height + second_cropped.height
    final_img = Image.new('RGB', (final_width, final_height))
    final_img.paste(first_cropped, (0, 0))
    final_img.paste(second_cropped, (0, first_cropped.height))

    output_filename = os.path.join(output_path, f"{base_name}.png")
    final_img.save(output_filename)
    print(f"Custom stitched image saved as '{output_filename}'")

if __name__ == "__main__":
    num_iterations = int(input("Enter how many times you want to run the screenshot capture: "))
    output_dir = input("Enter the output directory for screenshots (e.g., 'screenshots'): ").strip()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    for i in range(1, num_iterations + 1):
        # For each iteration, the base name is appended with the iteration number
        base_name = f"custom_stitched_{i}"
        print(f"\nStarting iteration {i} with base name '{base_name}'...")
        capture_and_stitch(output_dir, base_name)
        time.sleep(2)
