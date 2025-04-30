import os

# Define paths
txt_file_path = r"D:\University Work\Backup\emoji-detector-ai\data\emojis_unicode.txt"
label_file = r"D:\University Work\Backup\emoji-detector-ai\data\labels.txt"

# Load emojis and their Unicode values from the TXT file
emoji_unicode_mapping = []
with open(txt_file_path, "r", encoding="utf-8") as file:
    for line in file:
        emoji, unicode_value = line.strip().split(": ")
        emoji_unicode_mapping.append((emoji, unicode_value))

# Open file for writing labels
with open(label_file, 'w', encoding='utf-8') as f:
    for idx, (emoji, unicode_value) in enumerate(emoji_unicode_mapping, start=1):  # Start numbering from 1
        # Generate screenshot name based on the index
        screenshot_name = f"screenshot_{idx}.png"
        
        # Create the new file name based on the unicode value
        new_file_name = f"{unicode_value.replace(' ', '_')}.png"  # Replace spaces with underscores to form valid filenames

        # Write the label: new_file_name, emoji, unicode
        f.write(f'{new_file_name}, {emoji}, {unicode_value}\n')

print("✅ Labels file updated successfully with renamed image labels!")
