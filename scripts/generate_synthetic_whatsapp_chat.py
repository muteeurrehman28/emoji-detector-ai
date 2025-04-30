import os
import random
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta

# Constants
WIDTH, HEIGHT = 720, 1280
CHAT_HEADER_HEIGHT = 160
STATUS_BAR_HEIGHT = 40
FONT_PATH = "arial.ttf"
PROFILE_IMG_PATH = r"D:\University Work\Backup\emoji-detector-ai\assets\profile.png"
CHAT_BG_PATH = r"D:\University Work\Backup\emoji-detector-ai\assets\black_bg.jpeg"
target_directory = "assets/whatsapp_synthetic"
annotations_dir = os.path.join(target_directory, "annotations")
EMOJI_SIZE = 40  # Increased from 32 to match text height
MAX_BUBBLE_WIDTH = 504  # 70% of screen width

# Emoji configuration
emoji_dir = r"D:\University Work\Backup\emoji-detector-ai\emoji_data\cropped_emojis"
labels_file = r"D:\University Work\Backup\emoji-detector-ai\data\labels.txt"

# Colors (WhatsApp-accurate)
BG_COLOR = (236, 240, 241)
SENT_BUBBLE_COLOR = (231, 255, 219)
RECEIVED_BUBBLE_COLOR = (255, 255, 255)
HEADER_COLOR = (0, 168, 132)
STATUS_BAR_COLOR = (0, 148, 115)
TEXT_COLOR = (0, 0, 0)
TIMESTAMP_COLOR = (94, 110, 120)
TICK_COLOR_SENT = (94, 110, 120)
TICK_COLOR_READ = (0, 168, 255)
HIGHLIGHT_COLOR = (255, 255, 153)  # Light yellow for emoji highlight
BOX_COLOR = (255, 0, 0)  # Red for bounding box

# Create output directories
os.makedirs(target_directory, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Load emoji mappings with unicode
emoji_map = {}
emoji_unicode_map = {}
with open(labels_file, 'r', encoding='utf-8') as f:
    for line in f:
        filename, emoji, unicode = line.strip().split(', ', 2)  # Split on first two commas
        emoji_map[emoji] = os.path.join(emoji_dir, filename.strip())
        # Store the unicode as a single identifier (join multiple unicodes with underscores)
        unicode_identifier = unicode.replace(" ", "_")  # e.g., "U+1F468 U+1F3FD U+200D U+1F9B3" -> "U+1F468_U+1F3FD_U+200D_U+1F9B3"
        emoji_unicode_map[emoji] = unicode_identifier
emojis = list(emoji_map.keys())

# Required emojis for header/status bar
required_emojis = {
    'signal': '📶',  # U+1F4F6
    'battery': '🔋',  # U+1F50B
    'call': '📞',     # U+1F4DE
    'video': '📹'     # U+1F4F9
}
# Check for missing emojis
for key, emoji in required_emojis.items():
    if emoji not in emoji_map:
        print(f"Warning: Emoji {emoji} ({key}) not found in labels.txt. Using text fallback.")

# Sample messages
user_messages = [
    "Hi there!", "Can you send me the files?", "I’ll call you later.",
    "Thanks!", "No worries!", "Talk soon!", "Got it.", "Okay.", "Sure thing.", "Let me check.",
    "What's up?", "Ready for the meeting?", "Just arrived!", "See you soon.", "Awesome!"
]
bot_messages = [
    "Hello!", "Sure, I’ve sent the documents.", "Take care!", "Alright.",
    "No problem.", "Call me when you're free.", "You’re welcome!", "Yes, received.", "Cool.", "Done.",
    "Sounds good!", "On my way.", "Let me know!", "Perfect.", "Catch you later."
]
statuses = ["Sent", "Delivered", "Read"]

# Font handling
def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except IOError:
        print(f"Font {FONT_PATH} not found. Trying fallback.")
        for fallback in ["Arial.ttf", "SegoeUI.ttf", "LiberationSans-Regular.ttf"]:
            try:
                return ImageFont.truetype(fallback, size)
            except IOError:
                continue
        print("Using default font.")
        return ImageFont.load_default()

# Fonts
FONT_SIZE = 28
TIMESTAMP_FONT_SIZE = 14
HEADER_FONT_SIZE = 30
font = load_font(FONT_SIZE)
timestamp_font = load_font(TIMESTAMP_FONT_SIZE)
header_font = load_font(HEADER_FONT_SIZE)

def draw_bubble(draw, x, y, width, height, is_user):
    """Draw a WhatsApp-style message bubble with a tail and shadow."""
    radius = 12  # Matches screenshot
    bubble_color = SENT_BUBBLE_COLOR if is_user else RECEIVED_BUBBLE_COLOR
    # Shadow
    shadow_color = (200, 200, 200, 50)
    draw.rounded_rectangle(
        [x + 2, y + 2, x + width + 2, y + height + 2],
        radius=radius,
        fill=shadow_color,
        corners=(True, True, True, True)
    )
    # Main bubble
    draw.rounded_rectangle(
        [x, y, x + width, y + height],
        radius=radius,
        fill=bubble_color,
        corners=(True, True, True, True)
    )
    # Tail
    if is_user:
        tail = [
            (x + width - 12, y + height),
            (x + width, y + height),
            (x + width - 4, y + height - 12)
        ]
    else:
        tail = [
            (x, y + height),
            (x + 12, y + height),
            (x + 4, y + height - 12)
        ]
    draw.polygon(tail, fill=bubble_color)

def draw_status_ticks(draw, x, y, status):
    """Draw WhatsApp-style status ticks."""
    tick_color = TICK_COLOR_READ if status == "Read" else TICK_COLOR_SENT
    tick1 = [(x, y), (x + 4, y + 8), (x + 12, y - 4)]
    tick2 = [(x + 6, y), (x + 10, y + 8), (x + 18, y - 4)]
    draw.line(tick1, fill=tick_color, width=2)
    if status != "Sent":
        draw.line(tick2, fill=tick_color, width=2)

def draw_emoji(img, draw, x, y, emoji, size=EMOJI_SIZE, fallback_text=None, highlight=False):
    """Draw an emoji image with optional highlight and bounding box, return width and position."""
    if emoji in emoji_map:
        try:
            # Draw highlight background if requested
            if highlight:
                draw.rectangle([x - 2, y - 2, x + size + 2, y + size + 2], fill=HIGHLIGHT_COLOR)

            # Draw the emoji
            emoji_img = Image.open(emoji_map[emoji]).resize((size, size), Image.LANCZOS)
            img.paste(emoji_img, (int(x), int(y)), emoji_img if emoji_img.mode == 'RGBA' else None)

            # Draw bounding box if highlight is enabled
            if highlight:
                draw.rectangle([x, y, x + size, y + size], outline=BOX_COLOR, width=2)

            return size, (int(x), int(y), int(x) + size, int(y) + size), emoji_unicode_map.get(emoji, "unknown")
        except Exception as e:
            print(f"Error loading emoji {emoji}: {e}")
    # Fallback to text
    if fallback_text:
        draw.text((x, y), fallback_text, font=timestamp_font, fill=(255, 255, 255))
        return draw.textbbox((0, 0), fallback_text, font=timestamp_font)[2], None, None
    return 0, None, None

def wrap_text(draw, text, font, max_width):
    """Wrap text to fit within max_width, handling emojis."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + word + " "
        if word in emoji_map:
            test_width = draw.textbbox((0, 0), current_line, font=font)[2] + EMOJI_SIZE
        else:
            test_width = draw.textbbox((0, 0), test_line, font=font)[2]
        if test_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())
    return lines

def create_chat_image(filename):
    # Load background
    if os.path.exists(CHAT_BG_PATH):
        img = Image.open(CHAT_BG_PATH).resize((WIDTH, HEIGHT)).convert("RGB")
    else:
        img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)

    draw = ImageDraw.Draw(img)

    # List to store emoji positions
    emoji_positions = []

    # Status bar (no highlight for header emojis)
    draw.rectangle([0, 0, WIDTH, STATUS_BAR_HEIGHT], fill=STATUS_BAR_COLOR)
    draw.text((20, 10), "9:41", font=timestamp_font, fill=(255, 255, 255))
    width, pos, _ = draw_emoji(img, draw, WIDTH - 140, 10, required_emojis['signal'], size=20, fallback_text="LTE", highlight=False)
    if pos:
        emoji_positions.append((required_emojis['signal'], pos))
    width, pos, _ = draw_emoji(img, draw, WIDTH - 80, 10, required_emojis['battery'], size=20, fallback_text="85%", highlight=False)
    if pos:
        emoji_positions.append((required_emojis['battery'], pos))

    # Chat header (no highlight for header emojis)
    draw.rectangle([0, STATUS_BAR_HEIGHT, WIDTH, STATUS_BAR_HEIGHT + CHAT_HEADER_HEIGHT], fill=HEADER_COLOR)
    draw.polygon([(15, STATUS_BAR_HEIGHT + 80), (30, STATUS_BAR_HEIGHT + 65), (30, STATUS_BAR_HEIGHT + 95)], fill=(255, 255, 255))  # Back arrow
    # Profile picture
    if os.path.exists(PROFILE_IMG_PATH):
        profile = Image.open(PROFILE_IMG_PATH).resize((60, 60)).convert("RGBA")
        mask = Image.new("L", (60, 60), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((0, 0, 60, 60), fill=255)
        img.paste(profile, (50, STATUS_BAR_HEIGHT + 50), mask)
    else:
        draw.ellipse([50, STATUS_BAR_HEIGHT + 50, 110, STATUS_BAR_HEIGHT + 110], fill=(200, 200, 200))
    # Header text and icons
    draw.text((130, STATUS_BAR_HEIGHT + 60), "John Doe", font=header_font, fill=(255, 255, 255))
    draw.text((130, STATUS_BAR_HEIGHT + 100), "Online", font=timestamp_font, fill=(200, 255, 255))
    width, pos, _ = draw_emoji(img, draw, WIDTH - 130, STATUS_BAR_HEIGHT + 70, required_emojis['call'], size=30, fallback_text="☎", highlight=False)
    if pos:
        emoji_positions.append((required_emojis['call'], pos))
    width, pos, _ = draw_emoji(img, draw, WIDTH - 90, STATUS_BAR_HEIGHT + 70, required_emojis['video'], size=30, fallback_text="🎥", highlight=False)
    if pos:
        emoji_positions.append((required_emojis['video'], pos))
    draw.text((WIDTH - 50, STATUS_BAR_HEIGHT + 70), "⋮", font=header_font, fill=(255, 255, 255))

    y_offset = STATUS_BAR_HEIGHT + CHAT_HEADER_HEIGHT + 8
    current_time = datetime.now().replace(hour=9, minute=30)
    message_count = 0

    # Fill screen with messages
    while y_offset < HEIGHT - 40 and message_count < 50:
        is_user = random.choice([True, False])
        msg = random.choice(user_messages if is_user else bot_messages)
        if random.random() < 0.8:  # 80% chance to add emoji
            msg += " " + random.choice(emojis)
        time_stamp = (current_time + timedelta(minutes=random.randint(1, 5))).strftime("%I:%M %p").lstrip("0")
        status = random.choice(statuses)

        # Wrap text
        lines = wrap_text(draw, msg, font, MAX_BUBBLE_WIDTH - 40)
        text_width = 0
        text_height = 0
        for line in lines:
            line_width = 0
            for word in line.split():
                if word in emoji_map:
                    line_width += EMOJI_SIZE
                else:
                    line_width += draw.textbbox((0, 0), word + " ", font=font)[2]
            text_width = max(text_width, line_width)
            text_height += FONT_SIZE + 4

        # Calculate bubble width, ensuring space for timestamp and ticks
        timestamp_width = draw.textbbox((0, 0), time_stamp, font=timestamp_font)[2]
        bubble_width = text_width + 30
        if is_user:
            ticks_width = 20  # Approximate width of ticks
            bubble_width = max(bubble_width, text_width + timestamp_width + ticks_width + 40)
        else:
            bubble_width = max(bubble_width, text_width + timestamp_width + 30)
        bubble_height = text_height + 20
        bubble_width = min(bubble_width, MAX_BUBBLE_WIDTH)

        # Positioning
        x = WIDTH - bubble_width - 10 if is_user else 10

        # Check if bubble fits
        if y_offset + bubble_height > HEIGHT - 40:
            break

        # Draw bubble
        draw_bubble(draw, x, y_offset, bubble_width, bubble_height, is_user)

        # Draw text and emojis
        y_text = y_offset + 10
        for line in lines:
            x_text = x + 15
            for word in line.split():
                if word in emoji_map:
                    # Highlight and draw box around chat message emojis
                    width, pos, unicode = draw_emoji(img, draw, x_text, y_text - 6, word, highlight=True)
                    if pos:
                        emoji_positions.append((word, pos))
                    x_text += width + 4
                else:
                    draw.text((x_text, y_text), word + " ", font=font, fill=TEXT_COLOR)
                    x_text += draw.textbbox((0, 0), word + " ", font=font)[2]
            y_text += FONT_SIZE + 4

        # Timestamp and ticks
        time_x = x + bubble_width - timestamp_width - 10  # 10px from right edge
        time_y = y_offset + bubble_height - 6 - TIMESTAMP_FONT_SIZE  # 6px from bottom
        draw.text((time_x, time_y), time_stamp, font=timestamp_font, fill=TIMESTAMP_COLOR)
        if is_user:
            draw_status_ticks(draw, x + bubble_width - 30, time_y, status)

        y_offset += bubble_height + 8
        message_count += 1
        current_time += timedelta(minutes=random.randint(1, 5))

    # Save image
    img.save(filename, quality=95)
    print(f"Saved image: {filename}")

    # Save emoji positions
    annotation_filename = os.path.join(annotations_dir, f"chat_{os.path.basename(filename).split('_')[1].split('.')[0]}.txt")
    with open(annotation_filename, 'w', encoding='utf-8') as f:
        for emoji, (x1, y1, x2, y2) in emoji_positions:
            unicode = emoji_unicode_map.get(emoji, "unknown")
            if emoji in required_emojis.values():  # Skip header emojis
                continue
            f.write(f"{unicode} {x1} {y1} {x2} {y2}\n")
    print(f"Saved annotations: {annotation_filename}")

# Generate 100 images
if __name__ == "__main__":
    for i in range(100):
        file_path = os.path.join(target_directory, f"chat_{i+1:03}.png")
        create_chat_image(file_path)