"""Copy a Gemma-4 GGUF with the llama.cpp /v1/systemone metadata: decision type + "systemone" template.

  python gguf/add_metadata.py in.gguf out.gguf [--llama-cpp ~/llama.cpp] [--name "SynACK Decide 26B-A4B"]

Uses the gguf-py package from a llama.cpp checkout (needs a version with decision support, Oct 2026+).
Note: the output must be a different file from the input (macOS paths are case-insensitive).
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("input")
p.add_argument("output")
p.add_argument("--llama-cpp", default=str(Path.home() / "llama.cpp"), help="llama.cpp checkout (for gguf-py)")
p.add_argument("--name", default="SynACK Decide 26B-A4B")
p.add_argument("--type", default="nimble", help="decision type; nimble reads labels A-Z, AA, AB, ... (up to 255)")
a = p.parse_args()
if Path(a.input).resolve() == Path(a.output).resolve() or a.input.lower() == a.output.lower():
    sys.exit("output must be a different file from the input")

sys.path.insert(0, str(Path(a.llama_cpp).expanduser() / "gguf-py"))
import gguf  # noqa: E402
from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata, get_field_data  # noqa: E402

reader = gguf.GGUFReader(a.input, "r")
arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
S = gguf.GGUFValueType.STRING
meta = {
    gguf.Keys.General.NAME: MetadataDetails(S, a.name),
    gguf.Keys.Decision.TYPE.format(arch=arch): MetadataDetails(S, a.type),
    "tokenizer.chat_template.systemone": MetadataDetails(S, (HERE / "systemone.jinja").read_text()),
}
writer = gguf.GGUFWriter(a.output, arch=arch, endianess=reader.endianess)
alignment = get_field_data(reader, gguf.Keys.General.ALIGNMENT)
if alignment is not None:
    writer.data_alignment = alignment
copy_with_new_metadata(reader, writer, meta, [])
