import pyautogui
import time
import os
import pyperclip
import webbrowser

# Define paths
txt_file_path = r"D:\University Work\Semester IV\Software Engineering\Emoji Detector\emoji-detector-ai\emojis_unicode.txt"
screenshot_path = r"D:\University Work\Semester IV\Software Engineering\Emoji Detector\emoji-detector-ai\screenshots"
label_file = os.path.join(screenshot_path, 'labels.txt')

# Create folder if it doesn't exist
if not os.path.exists(screenshot_path):
    os.makedirs(screenshot_path)

# Define recipient phone number (Use country code, no spaces or dashes)
target_number = ""  # Replace with actual phone number

# Open WhatsApp Web with the target number
whatsapp_url = f"https://web.whatsapp.com/send?phone={target_number}"
webbrowser.open(whatsapp_url)  # Opens in the default web browser
time.sleep(15)  # Wait for WhatsApp Web to load

# Load emojis and their Unicode values from the TXT file
emoji_unicode_mapping = []
with open(txt_file_path, "r", encoding="utf-8") as file:
    for line in file:
        emoji, unicode_value = line.strip().split(": ")
        emoji_unicode_mapping.append((emoji, unicode_value))

# Open file for writing labels
with open(label_file, 'w', encoding='utf-8') as f:
    for idx, (emoji, unicode_value) in enumerate(emoji_unicode_mapping, start=1):
        # Copy emoji to clipboard
        pyperclip.copy(emoji)
        time.sleep(1)

        # Click message input box (Manually click once before running if needed)
        pyautogui.hotkey("ctrl", "v")  # Paste emoji
        time.sleep(1)

        # Press enter to send
        pyautogui.press("enter")
        time.sleep(2)  # Wait for the message to send

        # Take screenshot
        screenshot_file = os.path.join(screenshot_path, f'screenshot_{idx}.png')
        pyautogui.screenshot(screenshot_file)

        # Save label as "screenshot_name, emoji, unicode"
        f.write(f'{os.path.basename(screenshot_file)}, {emoji}, {unicode_value}\n')

print("✅ All emojis sent, labeled, and screenshots saved successfully!")
