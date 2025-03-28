import json

# Load the JSON file
file_path = r"D:\University Work\Semester IV\Software Engineering\emoji-detector-ai\data\whatsapp_emojis.json"

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)  # data is a list of dictionaries

# Function to get Unicode values for multi-character emojis
def get_unicode_values(emoji):
    return " ".join(f"U+{ord(char):X}" for char in emoji)

# Generate Unicode mappings for each emoji
emoji_unicode_mapping = {item["emoji"]: get_unicode_values(item["emoji"]) for item in data}

# Save to a TXT file
txt_file_path = r"D:\University Work\Semester IV\Software Engineering\emoji-detector-ai\data\emojis_unicode.txt"
with open(txt_file_path, "w", encoding="utf-8") as txt_file:
    for emoji, unicode_value in emoji_unicode_mapping.items():
        txt_file.write(f"{emoji}: {unicode_value}\n")

print(f"File saved successfully:\nTXT: {txt_file_path}")
