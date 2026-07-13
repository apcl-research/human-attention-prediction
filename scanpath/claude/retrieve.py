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
    1046788, 1118165, 11759898, 11950130, 1412807, 15689897, 16958722,
    1736289, 1810886, 18123253, 18251847, 18252350, 18420500, 18912425,
    18929060, 19280843, 19282261, 19344442, 19344536, 19346491, 19498298,
    20687719, 20950900, 21359951, 22407318, 22618655, 22622479, 22624602,
    22628734, 22907997, 24245709, 250694, 26412118, 26493872, 26501411,
    27798254, 27801498, 27802185, 27907979, 28953715, 2896279, 29852582,
    29854794, 29859244, 31203037, 31788771, 33519720, 33719114, 33719117,
    33719118, 33719607, 34413723, 34413807, 34413808, 34425716, 34426334,
    34426756, 34426938, 3456415, 3457090, 34609355, 35061399, 35553511,
    35553791, 37762493, 38184555, 38221537, 39215677, 3934822, 40099556,
    40776207, 40778768, 40865383, 40875567, 40879350, 4280405, 43040209,
    43040436, 43303607, 43419611, 4453291, 45888514, 45929468, 47479282,
    48104729, 48861766, 49121415, 51019251, 51122387
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