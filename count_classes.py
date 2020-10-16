from glob import glob

n_classes = 6
class_names = [
        "Crimson-Rumped Toucanet",
        "Scarlet Ibis",
        "Other Toucan",
        "Sunbittern",
        "Toco Toucan",
        "Inca Jay"
        ]

class_counters = [0, 0, 0, 0, 0, 0]

for filename in glob("./data/*.txt"):
    if "class" in filename:
        continue

    with open(filename, "r") as fp:
        for line in fp.readlines():
            class_index = int(line.split(" ")[0])
            if class_index >= n_classes:
                print(filename)
                continue
            class_counters[class_index] += 1

for i in range(n_classes):
    print(f"{class_names[i]}: {class_counters[i]}")

