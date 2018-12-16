import os
import json

PATH_DIR = "data/nav_drone/resources/synthetic_path_files"


def main():
    for fname in os.listdir(PATH_DIR):
        if fname.split(".")[-1] != "json":
            continue
        path = os.path.join(PATH_DIR, fname)
        with open(path) as f:
            path_json = json.load(f)
        for key in ("x_array", "z_array"):
            array = path_json[key]
            path_json[key] = array[:2] + 2 * [array[1]] + [array[-1]]

        with open(path, 'w') as f:
            json.dump(path_json, f)


if __name__ == "__main__":
    main()