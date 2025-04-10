import os
import shutil

# Paths
LABELS_FILE = r"D:\University Work\Semester IV\Software Engineering\emoji-detector-ai\screenshots\labels.txt"
PROCESSED_IMAGES_PATH = r"D:\University Work\Semester IV\Software Engineering\emoji-detector-ai\screenshots\processed"
CATEGORIZED_IMAGES_PATH = r"D:\University Work\Semester IV\Software Engineering\emoji-detector-ai\screenshots\categorized_images"

# Ensure categorized folder exists
os.makedirs(CATEGORIZED_IMAGES_PATH, exist_ok=True)

# Complete Unicode 15.1 Categorization System

UNICODE_CATEGORIES= {
    "Expressions": {  # formerly "Smileys & Emotion"
        "Faces": [
            (0x1F600, 0x1F64F),   # Emoticons block (e.g. 😀 to 🙏)
            (0x1F910, 0x1F92F),   # Supplemental Emoticons (e.g. 🤐 to 🤯)
            (0x1F973, 0x1F976),   # Recently added face emojis (🥳, 🥴, etc.)
            (0x1F97A, 0x1F97A),   # Specific emoji (e.g. 🥺)
            (0x1F9D0, 0x1F9D0),   # “Face with Monocle” emoji
            (0x2639, 0x263A),     # Classic text-based faces (☹, ☺)
            (0x1F31B, 0x1F31F),   # Additional face-like symbols (e.g. 🌛 to 🌟)
            (0x1FAE0, 0x1FAE8)    # Newer face emojis from Unicode 16.0
        ],
        "Hearts & Emotions": [
            (0x1F496, 0x1F49F),   # Heart-related emoji (e.g. 💖 to 💟)
            (0x1F5A4, 0x1F5A4),   # Black Heart (🖤)
            (0x1F90D, 0x1F90E),   # Additional heart faces (🤍, 🤎)
            (0x1F9E1, 0x1F9E1),   # Orange Heart style, etc.
            (0x2763, 0x2764),     # Heart symbols (❣, ❤)
            (0x1FAC0, 0x1FAC2),   # New heart designs from Unicode 16.0
            (0x1FA75, 0x1FA77)    # Additional heart-modifier sequences
        ],
        "Gestures & Body Language": [
            (0x1F44B, 0x1F450),   # Hand gestures (waving, ok sign, thumbs up, etc.)
            (0x1F485, 0x1F487),   # Facial/body care (e.g. 💅, 💇)
            (0x1F590, 0x1F595),   # Additional hand symbols (🖐, 🖖, etc.)
            (0x1F596, 0x1F5A4),   # Extended gestures (e.g. 🖖 followed by additional gesture sequences)
            (0x1F64B, 0x1F64F),   # Person gesturing (raising hand, bowing, etc.)
            (0x1F91F, 0x1F92F),   # Newer gesture emoji introduced in recent versions
            (0x1FAF0, 0x1FAF8),   # Updated gesture variants (from Unicode 16.0)
            (0x1F932, 0x1F932)    # Specific gesture emoji (e.g. 🤲)
        ]
    },
    "Human Representations": {  # formerly "People & Body"
        "Body Parts": [
            (0x1F440, 0x1F445),   # Eyes, ears, nose, mouth, etc.
            (0x1F4AA, 0x1F4AA),   # Flexed Biceps (💪)
            (0x1F9B0, 0x1F9B9),   # Hair, skin, and other body parts
            (0x1F9BB, 0x1F9BB),   # Additional body component
            (0x1F9BE, 0x1F9BF)    # Other body part symbols
        ],
        "Person Types": [
            (0x1F466, 0x1F478),   # Children and various adult figures
            (0x1F47C, 0x1F481),   # Baby, child, etc.
            (0x1F574, 0x1F575),   # Persons in suits, detective, etc.
            (0x1F57A, 0x1F57A),   # Person dancing, etc.
            (0x1F9D1, 0x1F9DF),   # General person emojis including gender variants
            (0x1F9AF, 0x1F9AF),   # Specific person type (e.g. person with headscarf)
            (0x1F9CD, 0x1F9CF),   # Family-related emojis (as base components)
            (0x1F9D4, 0x1F9D4)    # Person with beard emoji
        ],
        "Person Activities": [
            (0x1F46F, 0x1F46F),   # People holding hands (group activities)
            (0x1F93C, 0x1F93E),   # Sports and activities (e.g. wrestling, boxing)
            (0x1F9DE, 0x1F9DF),   # Advanced activity sequences
            (0x1F6C0, 0x1F6C0)    # Bath emoji (person bathing)
        ]
    },
    "Animals & Nature": {
        "Mammals": [
            (0x1F400, 0x1F43E),   # Range covering various mammal animals
            (0x1F98C, 0x1F9A2),   # Additional mammal emoji (e.g. 🦌 to 🦢)
            (0x1F9AB, 0x1F9AE)    # More mammal symbols (e.g. 🦫, 🦭)
        ],
        "Birds & Insects": [
            (0x1F426, 0x1F426),   # Specific bird (e.g. 🐦)
            (0x1F54A, 0x1F54A),   # Additional bird (e.g. 🕊) – note some systems may vary
            (0x1F985, 0x1F989),   # Range covering various birds (e.g. 🦅 to 🦉)
            (0x1F99C, 0x1F9A2)    # Insects and small animals – check for overlap with mammals
        ],
        "Marine Life": [
            # As of Unicode 16.0, many marine animal emoji are defined in Miscellaneous Symbols and Pictographs.
            (0x1F980, 0x1F984)    # Example range for marine life (e.g. 🦀, 🦞, 🦐, 🐙)
        ],
        "Dinosaurs": [
            (0x1F995, 0x1F996)    # Dinosaur emoji range
        ],
        "Mythical Creatures": [
            (0x1F409, 0x1F409),   # Dragon emoji
            (0x1F984, 0x1F984)    # Unicorn emoji
        ],
        "Plants & Flowers": [
            (0x1F330, 0x1F33F),   # Plant and tree emoji
            (0x1F340, 0x1F37F),   # Flowers and related symbols
            (0x1FAB0, 0x1FABF),   # New flower variants from Unicode 16.0
            (0x1F940, 0x1F945),   # Floral related range (e.g. 🥀 to 🥅)
            (0x1F950, 0x1F95E)    # Fruits and vegetables range
        ]
    },
    "Food & Beverage": {  # formerly "Food & Drink"
        "Fruits & Vegetables": [
            (0x1F345, 0x1F34F),   # Fruits range (e.g. 🍅 to 🍏)
            (0x1F951, 0x1F95F),   # Additional fruits and vegetables (e.g. 🥑 to 🥟)
            (0x1FAD0, 0x1FAD6),   # Recent additions for produce
            (0x1F95C, 0x1F96F)    # Extended produce range
        ],
        "Prepared Foods": [
            (0x1F32D, 0x1F37F),   # Foods like pizza, hot dog, burger, etc.
            (0x1F950, 0x1F950),   # Specific prepared food emoji (e.g. 🥐)
            (0x1F96F, 0x1F97F),   # Extended range for prepared foods
            (0x1F9C0, 0x1F9CB),   # Desserts and similar food items
            (0x1FAD0, 0x1FADF)    # Newer prepared food additions
        ],
        "Drinks & Tableware": [
            (0x1F964, 0x1F96C),   # Range including beverages (e.g. 🥤) and tableware
            (0x1F962, 0x1F963),   # Specific drinks (e.g. 🥢 sequences)
            (0x1F961, 0x1F961),   # Additional drink symbols
            (0x1F37B, 0x1F37C),   # Tableware symbols (e.g. 🍻)
            (0x1F942, 0x1F944)    # Extended tableware/drink items
        ]
    },
    "Travel & Geography": {  # formerly "Travel & Places"
        "Transportation": [
            (0x1F680, 0x1F6FF),   # Vehicles, airplanes, etc.
            (0x26F0, 0x26F5),     # Transport-related symbols (e.g. ⛰, ⛱)
            (0x1F6D1, 0x1F6D2),   # Specific transportation stops (e.g. 🛑)
            (0x1F6F3, 0x1F6F9),   # More transport symbols (e.g. 🚓 to 🚹)
            (0x1FA82, 0x1FA83)    # Recent additions to transportation symbols
        ],
        "Buildings & Infrastructure": [
            (0x1F3D4, 0x1F3F0),   # Buildings, castles, and landmarks
            (0x1F54B, 0x1F54E),   # Religious or historical buildings (e.g. 🕌 to 🕌 variants)
            (0x1F6D0, 0x1F6D2),   # Infrastructure symbols
            (0x1F9ED, 0x1F9EF),   # New building and infrastructure emoji
            (0x1F3E0, 0x1F3F0)    # Home and public buildings (duplicative with first range)
        ],
        "Astronomy & Weather": [
            (0x1F311, 0x1F31A),   # Moon phases, stars, etc.
            (0x2600, 0x2604),     # Weather-related symbols (sun, cloud, etc.)
            (0x26C5, 0x26C8),     # Partly cloudy, thunderstorms, etc.
            (0x1F324, 0x1F32F),   # Extended weather symbols (e.g. wind, tornado)
            (0x1F4AB, 0x1F4AB)    # Specific atmospheric phenomena (e.g. 💫)
        ]
    },
    "Activities & Sports": {  # formerly "Activities"
        "Sports": [
            (0x1F3C0, 0x1F3CF),   # Basketball, volleyball, etc.
            (0x1F3D0, 0x1F3DF),   # Sports equipment and venues
            (0x1F939, 0x1F93E),   # Dynamic sports (e.g. wrestling, boxing)
            (0x1F947, 0x1F94F),   # Medal and trophy sequences
            (0x1F94A, 0x1F94C)    # Additional sports symbols
        ],
        "Games & Toys": [
            (0x1F3AE, 0x1F3B3),   # Video games, arcade, etc.
            (0x1F9E9, 0x1F9EF),   # Toy and game-related emoji
            (0x1FA80, 0x1FA86),   # Modern games and toys (newer additions)
            (0x1FAA9, 0x1FAAC)    # Additional playful symbols
        ],
        "Cultural Activities": [
            (0x1F3A8, 0x1F3A8),   # Art palette (theater and performance)
            (0x1F3AD, 0x1F3AD),   # Performing arts (e.g. acting)
            (0x1F4F7, 0x1F4F7),   # Camera as part of cultural expression
            (0x1F9FF, 0x1F9FF)    # New cultural activity symbols
        ]
    },
    "Objects & Items": {  # formerly "Objects"
        "Clothing & Accessories": [
            (0x1F453, 0x1F45C),   # Headwear, eyewear, etc.
            (0x1F9E2, 0x1F9E5),   # Accessories like bags, etc.
            (0x1F9BA, 0x1F9BA),   # Specific accessory (e.g. 👺)
            (0x1FA70, 0x1FA74),   # Recent additions in clothing/accessory range
            (0x1FA78, 0x1FA7A)    # Additional accessory symbols
        ],
        "Technology & Tools": [
            (0x1F4BB, 0x1F4CF),   # Computers, phones, tablets, etc.
            (0x1F5A5, 0x1F5FA),   # Office and tech equipment
            (0x1F6E0, 0x1F6EC),   # Tools and mechanical devices
            (0x1F9F0, 0x1F9F9),   # Newer technological items (e.g. robots)
            (0x1FA9B, 0x1FA9F)    # Extended tools and tech symbols
        ],
        "Medical & Safety": [
            (0x1F489, 0x1F489),   # Syringe (💉)
            (0x1F48A, 0x1F48A),   # Pill (💊)
            (0x1F9F7, 0x1F9F9),   # Additional medical symbols (e.g. 🧷 variants)
            (0x1F6BD, 0x1F6BD),   # Toilet symbol often used in safety contexts
            (0x1F6CC, 0x1F6CC)    # Bed symbol (for hospital use)
        ],
        "Office & Stationery": [
            # As new office-related symbols are standardized, update these ranges.
            (0x1F4C0, 0x1F4C0),   # Example: Optical disc (📀) – placeholder
        ]
    },
    "Symbols & Signs": {  # formerly "Symbols"
        "Alphanumeric": [
            (0x0023, 0x0039),     # Digits and number sign (# through 9)
            (0x1F170, 0x1F189),   # Enclosed alphanumerics (A, B, etc.)
            (0x1F520, 0x1F52F)    # Miscellaneous alphanumerics
        ],
        "Geometric & Shapes": [
            (0x25AA, 0x25FF),     # Geometric shapes (small black squares, circles, etc.)
            (0x2B1B, 0x2B50),     # Additional shapes (e.g. large squares and stars)
            (0x1F7E0, 0x1F7EB)    # Colored circle and square emojis
        ],
        "Religious & Cultural": [
            (0x2620, 0x2626),     # Religious symbols (e.g. skull and crossbones, etc.)
            (0x2638, 0x2638),     # Specific cultural symbol (e.g. ☸)
            (0x267E, 0x267F),     # Additional religious signs
            (0x2695, 0x2695),     # Medical cross (often seen in religious contexts)
            (0x26CE, 0x26CE)      # Other cultural symbols
        ],
        "Warning & Status": [
            (0x26A0, 0x26A1),     # Warning signs (⚠, etc.)
            (0x26D3, 0x26D4),     # No entry and similar signs
            (0x1F6AB, 0x1F6AD),   # Prohibited signs (🚫 variants)
            (0x1F4DB, 0x1F4DB)    # Additional warning symbol
        ],
        "Zodiac": [
            (0x2648, 0x2653)      # Zodiac signs Aries through Pisces
        ]
    },
    "Flags": {
        "National Flags": [
            (0x1F1E6, 0x1F1FF)    # Regional Indicator Symbols, used in pairs for flags
        ],
        "Subdivision Flags": [
            (0x1F3F4, 0x1F3F4)    # Black flag with tag sequences for subdivisions
        ],
        "Special Flags": [
            (0x1F3F3, 0x1F3F3),   # White flag variants (🏳)
            (0x1F3C1, 0x1F3C1),   # Checkered flag (🏁)
            (0x1F6A9, 0x1F6A9)    # Triangular flag (🚩)
        ]
    },
    "Extras & Modifiers": {  # formerly "Extras"
        "Unicode Components": [
            (0x200D, 0x200D),     # Zero Width Joiner
            (0xFE0F, 0xFE0F),     # Variation Selector-16 (emoji presentation)
            (0x1F3FB, 0x1F3FF)    # Skin Tone Modifiers (Fitzpatrick types 1–6)
        ],
        "Keycaps": [
            (0x20E3, 0x20E3)      # Combining Enclosing Keycap
        ],
        "Other Symbols": [
            (0x1F004, 0x1F004),   # Mahjong Tile Red Dragon
            (0x1F0CF, 0x1F0CF)    # Playing Card Black Joker
        ]
    },
    "Finance & Business": {
        "Financial Symbols": [
            (0x1F4B0, 0x1F4B0),   # Money Bag
            (0x1F4B1, 0x1F4B1),   # Currency Exchange
            (0x1F4B2, 0x1F4B2),   # Heavy Dollar Sign
            (0x1F4B3, 0x1F4B3),   # Credit Card
            (0x1F4B8, 0x1F4B8)    # Money with Wings
        ]
    },
    "Media & Communication": {
        "Communication Devices": [
            (0x1F4F1, 0x1F4F1),   # Mobile Phone
            (0x260E,  0x260E),    # Telephone
            (0x1F4DE, 0x1F4DE)    # Telephone Receiver
        ],
        "Broadcast & Media": [
            (0x1F4FA, 0x1F4FA),   # Television
            (0x1F4FB, 0x1F4FB)    # Radio
        ]
    },
    "Science & Mathematics": {
        "Mathematical Operators": [
            (0x2200, 0x22FF)      # Common mathematical operators
        ],
        "Scientific Symbols": [
            (0x1F52C, 0x1F52C),   # Microscope
            (0x1F52D, 0x1F52D)    # Telescope
        ]
    },
    "Household & Daily Life": {
        "Furniture": [
            (0x1F6CB, 0x1F6CB)    # Couch
        ],
        "Kitchen & Utensils": [
            (0x1F373, 0x1F373)    # Cooking Pot
        ]
    },
    "Art & Literature": {
        "Books & Writing": [
            (0x1F4D6, 0x1F4D6),   # Open Book
            (0x1F4D5, 0x1F4D5)    # Closed Book
        ],
        "Theater & Performance": [
            (0x1F3AD, 0x1F3AD)    # Performing Arts Mask
        ]
    }
}

def get_category_and_subcategory(unicode_values):
    skin_tone_modifiers = range(0x1F3FB, 0x1F3FF + 1)
    for category, subcats in UNICODE_CATEGORIES.items():
        for subcategory, ranges in subcats.items():
            for unicode_value in unicode_values:
                try:
                    for code in unicode_value.split():
                        code_point = int(code.replace("U+", ""), 16)
                        if code_point in skin_tone_modifiers:
                            continue
                        for range_tuple in ranges:
                            start, end = range_tuple
                            if start <= code_point <= end:
                                return category, subcategory
                except ValueError:
                    print(f"⚠️ Invalid Unicode Value: {unicode_value}")
    return "Uncategorized", "Misc"

# Read labels file and store mappings
emoji_mapping = {}

with open(LABELS_FILE, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(", ")
        if len(parts) < 3:
            continue

        filename = parts[0]
        emoji = parts[1]
        unicode_values = parts[2:]

        emoji_mapping[filename] = {"emoji": emoji, "unicode": unicode_values}

print(f"📄 Loaded {len(emoji_mapping)} emoji mappings.")

# Process and copy images
for filename, data in emoji_mapping.items():
    source_path = os.path.join(PROCESSED_IMAGES_PATH, filename)

    if os.path.exists(source_path):
        category, subcategory = get_category_and_subcategory(data["unicode"])
        subcategory_path = os.path.join(CATEGORIZED_IMAGES_PATH, category, subcategory)
        os.makedirs(subcategory_path, exist_ok=True)
        shutil.copy2(source_path, os.path.join(subcategory_path, filename))
        print(f"✅ Copied {filename} → {subcategory_path}")
    else:
        print(f"⚠️ Image not found: {filename}")

print("🎉 Image categorization complete!")