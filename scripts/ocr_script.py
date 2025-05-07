import cv2
import pytesseract
import os
import requests
import re

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# URL for emoji-test.txt
EMOJI_URL = "https://unicode.org/Public/emoji/latest/emoji-test.txt"

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

def fetch_unicode_emojis(url):
    """Fetch and parse emoji characters from emoji-test.txt URL with custom categories."""
    emojis = set()
    try:
        response = requests.get(url)
        response.raise_for_status()
        lines = response.text.splitlines()
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):  # Skip comment lines
                parts = line.split(";")
                if len(parts) < 2:
                    continue
                # Extract emoji from the description part
                emoji_match = re.search(r'#\s*(\S+)', parts[1])
                if emoji_match:
                    emoji = emoji_match.group(1).split()[0]
                    if emoji:
                        emojis.add(emoji)
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch emoji data: {e}")
    return emojis

# Load emojis only once
EMOJI_SET = fetch_unicode_emojis(EMOJI_URL)

def remove_specific_emojis(text, emoji_set):
    """Remove only known emoji characters based on emoji-test.txt."""
    return ''.join(char for char in text if char not in emoji_set)

def extract_text_from_image(image_path):
    """Extract and clean text from an image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    text = pytesseract.image_to_string(gray, lang='eng')
    cleaned_text = remove_specific_emojis(text, EMOJI_SET)

    return cleaned_text.strip()

# Example usage with the uploaded image
image_path = r"D:\University Work\Backup\emoji-detector-ai\assets\Screenshot 2025-05-06 105302.png"  # Assume the image is saved as chat_screenshot.png

try:
    result = extract_text_from_image(image_path)
    print(result)
except Exception as e:
    print(f"Error: {e}")