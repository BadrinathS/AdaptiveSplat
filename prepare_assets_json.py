import argparse
import json
import re


def select_targets(context, target_count=None, target_stride=None, target_offset=1):
    if target_stride is not None:
        return context[target_offset::target_stride]

    if target_count is None:
        # Fallback: every alternate index if no guidance is provided.
        return context[0::2]

    step = max(1, len(context) // target_count)
    return context[target_offset::step][:target_count]

def parse_log(file_path, target_count=None, target_stride=None, target_offset=1):
    result = {}

    pattern = re.compile(
        r"validation step\s+\d+;\s*scene\s*=\s*\['([^']+)'\];\s*context\s*=\s*\[\[([^\]]+)\]\]"
    )

    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                scene = match.group(1)
                if not scene.startswith("dl3dv_"):
                    continue
                scene = scene.split('_')[1]
                context_str = match.group(2)

                context = [int(x.strip()) for x in context_str.split(",")]

                target = select_targets(
                    context,
                    target_count=target_count,
                    target_stride=target_stride,
                    target_offset=target_offset,
                )

                result[scene] = {
                    "context": context,
                    "target": target
                }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-file",
        required=True,
        help="Validation log to parse (lines of the form "
             "\"validation step N; scene = ['dl3dv_<id>']; context = [[i, j, ...]]\").",
    )
    parser.add_argument(
        "--out-file",
        required=True,
        help="Path of the context/target index JSON to write.",
    )
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--target-stride", type=int, default=8)
    parser.add_argument("--target-offset", type=int, default=1)
    args = parser.parse_args()

    data = parse_log(
        args.log_file,
        target_count=args.target_count,
        target_stride=args.target_stride,
        target_offset=args.target_offset,
    )

    with open(args.out_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Parsed {len(data)} scenes")