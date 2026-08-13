import os
import glob

label_folders = ["train/labels", "valid/labels", "test/labels"]

print("Starting to merge classes...")

for folder in label_folders:
    if not os.path.exists(folder):
        print(f"Skipping {folder} (Not found)")
        continue

    txt_files = glob.glob(os.path.join(folder, "*.txt"))
    
    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            old_class = int(parts[0])
            
            # 0, 1, 2 become Helmet (0). 3 becomes No_Helmet (1).
            if old_class in [0, 1, 2]:
                new_class = 0
            elif old_class == 3:
                new_class = 1
            else:
                continue 

            parts[0] = str(new_class)
            new_lines.append(" ".join(parts) + "\n")
            
        with open(txt_file, "w") as f:
            f.writelines(new_lines)

print("Success! All labels have been converted to 2 classes.")