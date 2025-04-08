import os
from tqdm import tqdm
import shutil


def create_test_images(input_path, output_path):

    with open(os.path.join(input_path, "test.ln"), "r") as f:
        lines = f.readlines()

    empty_counter = 0
    
    for line in tqdm(lines):
        file_name = line.strip()
        file_path = os.path.join(input_path, "lines", file_name)
        txt_path = os.path.join(input_path, "lines", file_name.replace(".jpg", ".txt"))

        label = open(txt_path, "r").read()

        if label == "":
            empty_counter += 1
            label = f"empty_{empty_counter}"

        dest_path = os.path.join(output_path, label + ".jpg")
        shutil.copy(file_path, dest_path)

def replace_X_with_nothing_everywhere(input_path):
    for file in os.listdir(input_path):
        new_lines = []
        if file.endswith(".txt"):
            with open(os.path.join(input_path, file), "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.replace("X", "")
                new_lines.append(line)

            with open(os.path.join(input_path, file), "w") as f:
                f.write("\n".join(new_lines))

if __name__ == "__main__":
    input_path = "/workspace/HTR-VT/data/custom_dataset/"
    output_path = "/workspace/test_images"
    os.makedirs(output_path, exist_ok=True)
    create_test_images(input_path, output_path)
    # replace_X_with_nothing_everywhere(os.path.join(input_path, "lines"))


