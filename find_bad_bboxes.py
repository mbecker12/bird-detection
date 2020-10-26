from glob import glob

for filename in glob("data/train/*.txt"):
    with open(filename, "r") as fp:
        for line in fp.readlines():
            try:
                _, x, y, width, height = line.split(" ")
            except:
                print(f"{line=}")
            x = float(x)
            y = float(y)
            width = float(width)
            height = float(height)
            x_min = x - width * 0.5
            x_max = x + width * 0.5
            y_min = y - height * 0.5
            y_max = y + height * 0.5

            try:
                assert x_min > 0, filename
                assert y_min > 0, filename
                assert x_max < 1, filename
                assert y_max < 1, filename
            except:
                print(line.replace("\n",""))
                print(f"{filename=}: {x_min=}, {x_max=}, {y_min=}, {y_max=}")
                print()
            