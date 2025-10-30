# gen_mats_to_file.py
import sys
import numpy as np
from pathlib import Path

def cpp_array(name: str, ctype: str, arr: np.ndarray) -> str:
    flat = arr.flatten()
    values = ", ".join(str(int(x)) for x in flat)
    return f"{ctype} {name}[] = {{{values}}};"

def main():
    if len(sys.argv) != 2:
        print("Usage: python gen_mats_to_file.py <output_file_path>")
        sys.exit(1)

    out_path = Path(sys.argv[1])

    np.random.seed(42)

    # Generate A in {-1, 0, 1}
    A = np.random.choice([-1, 0, 1], size=(64, 64)).astype(np.int8)

    # Generate B in {-1, 1}
    B = np.random.choice([-1, 1], size=(64, 64)).astype(np.int8)

    # Multiply 
    C = np.sign(A.astype(np.int8) @ B.astype(np.int8)) # 64x64 int8

    # Compose C++ content
    header = "#include <cstdint>\n\n"
    content = [
        cpp_array("A", "int8_t", A),
        "",
        cpp_array("B", "int8_t", B),
        "",
        cpp_array("C", "int8_t", C),
        ""
    ]
    text = header + "\n".join(content) + "\n"

    # Write to file
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote matrices to: {out_path}")

if __name__ == "__main__":
    main()
