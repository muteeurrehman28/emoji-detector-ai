# sscapturing.py

import io
import math
import os
import time
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from PIL import Image

def capture_and_stitch(output_path, base_name):
    # your new crop values
    start_x, end_x = 675, 1215

    # point to your local msedgedriver.exe
    EDGE_DRIVER_PATH = r"D:\edgedriver_win64\msedgedriver.exe"
    service = Service(executable_path=EDGE_DRIVER_PATH)
    driver  = webdriver.Edge(service=service)

    # set window size and load the page with base_name
    driver.set_window_size(1920, 1080)
    driver.get(f"http://127.0.0.1:5000/?base_name={base_name}")
    time.sleep(3)  # wait for JS to generate & save annotations

    # measure full page
    total_h     = driver.execute_script("return document.body.scrollHeight")
    view_h      = driver.execute_script("return window.innerHeight")
    num_screens = math.ceil(total_h / view_h)

    # capture each viewport
    shots = []
    for i in range(num_screens):
        y = total_h - view_h if i == num_screens - 1 else i * view_h
        driver.execute_script("window.scrollTo(0, arguments[0]);", y)
        time.sleep(1)
        png = driver.get_screenshot_as_png()
        shots.append(Image.open(io.BytesIO(png)))

    driver.quit()

    if len(shots) < 2:
        print("Need at least 2 screenshots to stitch.")
        return

    # crop first shot (from y=30 down)
    top_crop = shots[0].crop((start_x, 30, end_x, shots[0].height))

    # crop second shot bottom ~38.9%
    h1 = shots[1].height - 30
    y1 = int(h1 * 0.6210)
    bot_crop = shots[1].crop((start_x, y1, end_x, h1))

    # stitch together
    w = end_x - start_x
    H = top_crop.height + bot_crop.height
    out = Image.new("RGB", (w, H))
    out.paste(top_crop, (0, 0))
    out.paste(bot_crop, (0, top_crop.height))

    # save to output_path/<base_name>.png
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, f"{base_name}.png")
    out.save(out_file)
    print("Saved image:", out_file)


if __name__ == "__main__":
    # 1) Ask for starting numeric suffix
    start_num = int(input("Enter starting number (e.g. '90'): ").strip())
    # 2) Ask how many captures
    count     = int(input("How many captures? ").strip())
    # 3) Ask for output directory
    out_dir   = input("Screenshots folder: ").strip()

    for i in range(count):
        current   = start_num + i
        base_name = f"custom_stitched_{current}"
        print(f"\nIteration {i+1}, saving as '{base_name}.png'")
        capture_and_stitch(out_dir, base_name)
        time.sleep(2)
