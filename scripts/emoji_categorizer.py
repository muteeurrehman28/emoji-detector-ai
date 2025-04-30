import requests
import os
import shutil

# URL to latest emoji-test.txt
url = "https://unicode.org/Public/emoji/latest/emoji-test.txt"

# Custom category mapping
CUSTOM_CATEGORY_MAPPING = {
    "Smileys & Emotion": "Expressions & Faces",
    "People & Body": "People & Body",
    "Component": "Modifiers Gondi & Extras",
    "Animals & Nature": "Animals",
    "Food & Drink": "Food & Drinks",
    "Travel & Places": "Transport & Travel",
    "Activities": "Sports & Games",
    "Objects": "Technology & Devices",
    "Symbols": "Signs & Symbols",
    "Flags": "Flags",
    "person": "People & Body",
    "person-role": "People & Body",
    "person-fantasy": "People & Body",
    "person-dress": "People & Body",
    "person-resting": "People & Body",
    "person-sport": "Sports & Games",
    "person-activity": "Sports & Games",
    "person-gesture": "Hand Gestures",
    "hand-fingers-open": "Hand Gestures",
    "hand-fingers-partial": "Hand Gestures",
    "hand-single-finger": "Hand Gestures",
    "hand-fingers-closed": "Hand Gestures",
    "hands": "Hand Gestures",
    "person-hair": "Hair & Appearance",
    "body-parts": "Body Parts",
    "family": "Family & Relationships",
    "animal-mammal": "Animals",
    "animal-bird": "Birds & Insects",
    "animal-bug": "Birds & Insects",
    "animal-marine": "Animals",
    "animal-amphibian": "Animals",
    "plant-flower": "Nature & Plants",
    "plant-other": "Nature & Plants",
    "sky & weather": "Weather & Sky",
    "place-building": "Buildings & Places",
    "place-religious": "Buildings & Places",
    "place-geographic": "Buildings & Places",
    "place-map": "Buildings & Places",
    "place-other": "Buildings & Places",
    "hotel": "Buildings & Places",
    "transport-ground": "Transport & Travel",
    "transport-water": "Transport & Travel",
    "transport-air": "Transport & Travel",
    "food-fruit": "Food & Drinks",
    "food-vegetable": "Food & Drinks",
    "food-prepared": "Food & Drinks",
    "food-sweet": "Food & Drinks",
    "drink": "Food & Drinks",
    "dishware": "Kitchen & Household",
    "clothing": "People & Body",
    "money": "Money & Finance",
    "medical": "Medical & Health",
    "music": "Music & Instruments",
    "musical-instrument": "Music & Instruments",
    "sound": "Music & Instruments",
    "event": "Celebration & Events",
    "award-medal": "Celebration & Events",
    "game": "Sports & Games",
    "arts & crafts": "Art & Creativity",
    "writing": "Art & Creativity",
    "mail": "Mail & Communication Symbols",
    "communication": "Mail & Communication Symbols",
    "time": "Time & Clocks",
    "love": "Love & Emotions",
    "emotion": "Love & Emotions",
    "zodiac": "Zodiac & Astrology",
    "keycap": "Emoji Keycaps & Digits",
    "digit": "Emoji Keycaps & Digits",
    "math": "Signs & Symbols",
    "punctuation": "Signs & Symbols",
    "currency": "Money & Finance"
}

# Function to convert Unicode (e.g., "1F600" or "1F468 200D 1F469") to emoji
def unicode_to_emoji(unicode_str):
    # Split multiple code points
    code_points = unicode_str.split()
    try:
        # Convert each code point to a character
        chars = []
        for cp in code_points:
            cp_int = int(cp, 16)
            # Ensure code point is in valid range (0x0 to 0x10FFFF)
            if 0 <= cp_int <= 0x10FFFF:
                chars.append(chr(cp_int))
            else:
                print(f"Warning: Code point {cp} out of valid range")
                return None
        return "".join(chars)
    except (ValueError, OverflowError) as e:
        print(f"Error converting Unicode {unicode_str}: {e}")
        return None

# Download and parse emoji-test.txt
response = requests.get(url)
lines = response.text.splitlines()

emoji_to_custom_category = {}
current_group = ""
current_subgroup = ""

for line in lines:
    line = line.strip()
    if line.startswith("# group:"):
        current_group = line.split(":")[1].strip()
    elif line.startswith("# subgroup:"):
        current_subgroup = line.split(":")[1].strip()
    elif line and not line.startswith("#"):
        try:
            # Extract Unicode sequence (before first semicolon)
            parts = line.split(";")
            unicode_seq = parts[0].strip()  # e.g., "1F600" or "1F468 200D 1F469"
            emoji = unicode_to_emoji(unicode_seq)
            if not emoji:
                continue
            # Map using subgroup first, then group, else "Uncategorized"
            custom_category = (
                CUSTOM_CATEGORY_MAPPING.get(current_subgroup)
                or CUSTOM_CATEGORY_MAPPING.get(current_group)
                or "Uncategorized"
            )
            emoji_to_custom_category[emoji] = custom_category
        except (IndexError, ValueError) as e:
            print(f"Error parsing line: {line} - {e}")
            continue

# Directory containing emoji images
image_dir = r"D:\University Work\Backup\emoji-detector-ai\emoji_data\processed"
output_dir = r"D:\University Work\Backup\emoji-detector-ai\emoji_data\green-bg-classified-emojis"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Track created folders and unmapped images
created_folders = set()
unmapped_images = []

# Function to normalize Unicode filename (e.g., "U+1F600" to "1F600")
def normalize_unicode_filename(filename):
    return filename.replace("U+", "").replace(".png", "").replace("_", " ")

# Process each image in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        # Extract and normalize Unicode from filename (e.g., "U+1F600.png" or "U+1F468_U+200D_U+1F469.png")
        unicode_str = normalize_unicode_filename(filename)
        emoji = unicode_to_emoji(unicode_str)
        
        if emoji and emoji in emoji_to_custom_category:
            category = emoji_to_custom_category[emoji]
            # Create category folder if not exists
            category_folder = os.path.join(output_dir, category)
            if category not in created_folders:
                os.makedirs(category_folder, exist_ok=True)
                created_folders.add(category)
            
            # Copy the image to the category folder
            src_path = os.path.join(image_dir, filename)
            dst_path = os.path.join(category_folder, filename)
            shutil.copy2(src_path, dst_path)
        else:
            unmapped_images.append(filename)

# Print summary
print(f"Total images processed: {len(os.listdir(image_dir))}")
print(f"Images successfully classified: {len(os.listdir(image_dir)) - len(unmapped_images)}")
print(f"Categories created: {len(created_folders)}")
print(f"Unmapped images: {len(unmapped_images)}")
if unmapped_images:
    print("Sample unmapped images:", unmapped_images[:5])
print(f"Classified emojis saved in: {output_dir}")