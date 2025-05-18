import os
import json

def rebase_annotations(input_dir: str,
                       output_dir: str,
                       offset_x: float = 0,
                       offset_y: float = 0,
                       dpr: float = 1.0,
                       clamp: bool = True):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.json'):
            continue

        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        with open(in_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        updated_anns = []
        for ann in data.get("annotations", []):
            x, y, w, h = ann["bbox"]

            # Rebase + scale
            x_new = (x - offset_x) * dpr
            y_new = (y - offset_y) * dpr
            w_new = w * dpr
            h_new = h * dpr

            # Round & optionally clamp
            x_new = max(0, round(x_new)) if clamp else round(x_new)
            y_new = max(0, round(y_new)) if clamp else round(y_new)
            w_new = round(w_new)
            h_new = round(h_new)

            new_ann = ann.copy()
            new_ann["bbox"] = [x_new, y_new, w_new, h_new]
            updated_anns.append(new_ann)

        data["annotations"] = updated_anns

        # ✅ Scale image width/height if present
        if "width" in data:
            data["width"] = round(data["width"] * dpr)
        if "height" in data:
            data["height"] = round(data["height"] * dpr)

        # Save the rebased JSON
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        print(f"✔ Processed: {filename} (scaled with DPR={dpr})")

# ──────────────────────────────────────────────────────────────
# ✅ Example usage in Google Colab or your script



input_dir = '/content/drive/MyDrive/emoji_dataset/2_annotation/annotation_data2'
output_dir = '/content/drive/MyDrive/emoji_dataset/2_annotations_rebased'

# Example values based on your setup
offset_x = 448           # Chat container's left offset in CSS px
offset_y = 20            # Chat container's top offset in CSS px
dpr = 1.5              # Your device pixel ratio from browser
clamp = True             # Clamp negative coordinates to 0

rebase_annotations(input_dir, output_dir, offset_x, offset_y, dpr, clamp)
