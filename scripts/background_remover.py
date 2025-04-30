from rembg import remove
import os

def remove_background(input_path, output_path):
    # Open the image file
    with open(input_path, 'rb') as input_file:
        input_data = input_file.read()

    # Remove the background
    output_data = remove(input_data)

    # Write the output (transparent background image)
    with open(output_path, 'wb') as output_file:
        output_file.write(output_data)

    print(f"Background removed. Saved: {output_path}")

def process_images_in_directory(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, f'{filename}')
            remove_background(input_path, output_path)

# Set your directories here
input_directory = r'D:\University Work\Backup\emoji-detector-ai\emoji_data\processed'
output_directory = r'D:\University Work\Backup\emoji-detector-ai\emoji_data\bg-removed'

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

process_images_in_directory(input_directory, output_directory)
