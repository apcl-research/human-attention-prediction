from anthropic import Anthropic
from tqdm import tqdm
import argparse
import os

# =========================
# Arguments
# =========================
parser = argparse.ArgumentParser(description="Download Claude batch results.")

parser.add_argument(
    "--batch_id",
    type=str,
    required=True,
    help="Anthropic batch ID."
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="scanpath_predictions_claude_incontext_5tokens",
    help="Directory to save prediction files."
)

parser.add_argument(
    "--key_file",
    type=str,
    default="/home/chiayi/claude.key",
    help="Path to Anthropic API key file."
)

args = parser.parse_args()

# =========================
# Load API key
# =========================
with open(args.key_file, "r") as f:
    key = f.readline().strip()

client = Anthropic(api_key=key)

# Create output directory if needed
os.makedirs(args.output_dir, exist_ok=True)

fid_list = [
    10222579, 14467418, 17121898, 18418213, 19498280, 19682824,
    26285656, 29601536, 33719869, 36405409, 39120328, 39299426,
    4114383, 45047585, 4627680, 49250848, 50994916, 12723449,
    14477536, 1782360, 18421665, 19505695, 20787007, 28631042,
    31696447, 34105249, 36634895, 39233258, 39840471, 41287529,
    45130358, 47570692, 49866815, 50995324, 12725774, 16777940,
    1810081, 19218425, 19507414, 23014476, 29318894, 31789275,
    34427273, 38221424, 39233866, 40865212, 43137003, 45147874,
    47571713, 50026101, 51577053, 13482891, 1694531, 18354735,
    19413001, 19507735, 26215341, 29572299, 33718481, 34604973,
    38737384, 39298970, 40936756, 44521080, 46026508, 47571922,
    50891793, 7957602
]

for fid in tqdm(fid_list):
    output_file = os.path.join(args.output_dir, f"predict_{fid}.txt")

    with open(output_file, "w") as pf:
        for result in client.messages.batches.results(args.batch_id):
            result = result.model_dump()
            function_id = result["custom_id"]

            if str(fid) not in str(function_id):
                continue

            try:
                scanpath = result["result"]["message"]["content"][0]["text"]
                pf.write(f"{function_id}\t{scanpath}\n")
            except Exception:
                pf.write(f"{function_id}\terror\n")
