import subprocess
import pickle
from tqdm import tqdm
import argparse




# =========================================================
# 1. srcML Parsing: Code → AST Nodes + Sparse Adjacency
# =========================================================
def code_to_srcml(code: str, language: str = "Java") -> str:
    """
    Convert source code into srcML XML string.
    Requires srcml to be installed (https://www.srcml.org).
    """
    process = subprocess.Popen(
        ["srcml", "--language", language],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True
    )
    xml_output, error = process.communicate(code)
    if process.returncode != 0:
        raise RuntimeError(f"srcML failed: {error}")
    return xml_output





if __name__ == "__main__":
    # =========================================================
    # Arguments: input & output paths
    # =========================================================
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input fixation data pickle file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output fixation data pickle file"
    )
    args = parser.parse_args()

    fixation_data = pickle.load(open(args.input_path, "rb"))


    new_fixation_data = {}
    for key in tqdm(list(fixation_data.keys())[:]):
        temp_new_data = []
        alldata = fixation_data[key]
        for data in alldata:
            function = data["function"]
            xml_str = code_to_srcml(function, language="Java")
            data["srcml"] = xml_str
            temp_new_data.append(data)
        new_fixation_data[key] = temp_new_data
    print(len(list(new_fixation_data.keys())))
    print(new_fixation_data.keys())






    pickle.dump(new_fixation_data, open(args.output_path, "wb"))
