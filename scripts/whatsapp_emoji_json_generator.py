import emoji
import json
import os

# Define the directory to save the JSON file
save_dir = r"D:\University Work\Semester IV\Software Engineering\emoji-detector-ai\data"
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, "whatsapp_emojis.json")

# Get all emoji data
all_emojis = emoji.EMOJI_DATA

# Create a structured dataset with details for each emoji
emoji_list = []
for char, data in all_emojis.items():
    emoji_list.append({
        "emoji": char,
        "short_names": data.get("short_names", []),
        "category": data.get("category", ""),
        "unicode": data.get("unicode", "")
    })

# Save to a JSON file
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(emoji_list, f, ensure_ascii=False, indent=4)

print(f"WhatsApp emojis JSON file created successfully at: {file_path}")