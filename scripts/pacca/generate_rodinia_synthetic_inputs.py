#!/usr/bin/env python3
"""Genera los datos de entrada sinteticos que necesitan 3 de los kernels
Rodinia del catalogo (ARC-75): myocyte (y.txt/params.txt), dwt2d (BMP) y
heartwall (AVI). El mirror de Rodinia (github.com/yuhc/gpu-rodinia) trae
data/ vacio (solo .gitkeep); estos generadores reemplazan esos datasets
sin necesitar herramientas externas (ffmpeg no esta disponible en el nodo
de computo paccaA100, solo en el login node).

No requiere GPU ni el binario del kernel -- solo Python estandar. Uso:

    python generate_rodinia_synthetic_inputs.py --out-root ~/hyperion-kernels/data

Escribe:
    <out-root>/myocyte/y.txt
    <out-root>/myocyte/params.txt
    <out-root>/dwt2d/test_192.bmp
    <out-root>/heartwall/test.avi
"""
import argparse
import random
import struct
from pathlib import Path

# myocyte: Rodinia lee 91 valores de estado inicial (y.txt) y 18
# parametros (params.txt), uno por linea -- ver work.cu/work_2.cu.
MYOCYTE_Y_COUNT = 91
MYOCYTE_PARAMS_COUNT = 18


def write_myocyte_inputs(out_dir: Path, seed: int = 1) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_values = [rng.uniform(0.0, 1.0) for _ in range(MYOCYTE_Y_COUNT)]
    param_values = [rng.uniform(0.0, 1.0) for _ in range(MYOCYTE_PARAMS_COUNT)]
    (out_dir / "y.txt").write_text("\n".join(f"{v:.6f}" for v in y_values) + "\n")
    (out_dir / "params.txt").write_text("\n".join(f"{v:.6f}" for v in param_values) + "\n")


def write_bmp(path: Path, width: int, height: int, seed: int = 1) -> None:
    """BMP de 24 bits (3 canales), sin compresion -- formato que dwt2d
    espera con --components 3 (el valor por defecto)."""
    rng = random.Random(seed)
    row_size = (width * 3 + 3) & ~3  # cada fila alineada a 4 bytes
    pixel_data_size = row_size * height
    file_size = 54 + pixel_data_size

    header = struct.pack(
        "<2sIHHI IiiHHIIiiII",
        b"BM", file_size, 0, 0, 54,
        40, width, height, 1, 24, 0, pixel_data_size, 0, 0, 0, 0,
    )

    rows = []
    for _ in range(height):
        row = bytes(rng.randrange(256) for _ in range(width * 3))
        row += b"\x00" * (row_size - len(row))
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        for row in rows:
            f.write(row)


def _chunk(fourcc: bytes, data: bytes) -> bytes:
    padded = data + (b"\x00" if len(data) % 2 else b"")
    return fourcc + struct.pack("<I", len(data)) + padded


def write_avi(path: Path, width: int, height: int, n_frames: int, seed: int = 1) -> None:
    """AVI RIFF minimo (video sin comprimir, 8 bits grises) -- reemplaza
    a ffmpeg, que no esta disponible en el nodo de computo paccaA100."""
    rng = random.Random(seed)
    frame_size = width * height
    frames = [bytes(rng.randrange(256) for _ in range(frame_size)) for _ in range(n_frames)]

    avih = struct.pack(
        "<14I",
        1_000_000 // 25, frame_size * 25, 0, 0x10, n_frames, 0, 1,
        frame_size, width, height, 0, 0, 0, 0,
    )
    strh = struct.pack(
        "<4s4sIHHIIIIIIIIhhhh",
        b"vids", b"DIB ", 0, 0, 0, 0, 1, 25, 0, n_frames,
        frame_size, 0xFFFFFFFF, frame_size, 0, 0, width, height,
    )
    bmih = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 8, 0, frame_size, 0, 0, 256, 0)
    palette = b"".join(struct.pack("<4B", i, i, i, 0) for i in range(256))
    strf = bmih + palette

    strl_body = b"strl" + _chunk(b"strh", strh) + _chunk(b"strf", strf)
    strl = b"LIST" + struct.pack("<I", len(strl_body)) + strl_body
    hdrl_body = b"hdrl" + _chunk(b"avih", avih) + strl
    hdrl = b"LIST" + struct.pack("<I", len(hdrl_body)) + hdrl_body

    movi_body = b"movi" + b"".join(_chunk(b"00db", f) for f in frames)
    movi = b"LIST" + struct.pack("<I", len(movi_body)) + movi_body

    idx_entries = []
    offset = 4
    for f in frames:
        idx_entries.append(struct.pack("<4sIII", b"00db", 0x10, offset, len(f)))
        offset += 8 + len(f) + (len(f) % 2)
    idx1 = _chunk(b"idx1", b"".join(idx_entries))

    riff_body = b"AVI " + hdrl + movi + idx1
    riff = b"RIFF" + struct.pack("<I", len(riff_body)) + riff_body

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(riff)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    # ARC-86: parametrizado para poder alargar heartwall (mas cuadros a
    # procesar, sin cambiar resolucion) y dwt2d (imagen mas grande, mismo
    # kernel fdwt97Kernel por pixel) al re-caracterizar warmup real.
    parser.add_argument("--heartwall-frames", type=int, default=25)
    parser.add_argument("--dwt2d-size", type=int, default=192)
    # ARC-89: por defecto, este script SIEMPRE reescribia los 3 archivos en
    # cada invocacion -- una llamada para regenerar solo uno (por ejemplo,
    # dwt2d con --dwt2d-size distinto) silenciosamente devolvia heartwall/
    # myocyte a los defaults (25 cuadros / semilla fija), incluso si ya
    # habian sido alargados a proposito para una caracterizacion previa.
    # Encontrado en produccion: el test.avi de 20000 cuadros generado para
    # ARC-86 volvio a 25 cuadros en algun punto posterior sin que ningun
    # cambio de catalogo lo pidiera. Por defecto ahora se salta cualquier
    # archivo que ya exista -- --force reproduce el comportamiento viejo
    # (siempre reescribir) para cuando de verdad se quieren regenerar los
    # tres a la vez.
    parser.add_argument("--force", action="store_true",
                         help="Reescribe los 3 archivos aunque ya existan (comportamiento previo a ARC-89).")
    args = parser.parse_args()

    myocyte_dir = args.out_root / "myocyte"
    dwt2d_path = args.out_root / "dwt2d" / f"test_{args.dwt2d_size}.bmp"
    heartwall_path = args.out_root / "heartwall" / "test.avi"

    if args.force or not (myocyte_dir / "y.txt").exists():
        write_myocyte_inputs(myocyte_dir, seed=args.seed)
    else:
        print(f"Omitido (ya existe, usar --force para regenerar): {myocyte_dir / 'y.txt'}")

    if args.force or not dwt2d_path.exists():
        write_bmp(dwt2d_path, args.dwt2d_size, args.dwt2d_size, seed=args.seed)
    else:
        print(f"Omitido (ya existe, usar --force para regenerar): {dwt2d_path}")

    if args.force or not heartwall_path.exists():
        write_avi(heartwall_path, 128, 128, args.heartwall_frames, seed=args.seed)
    else:
        print(f"Omitido (ya existe, usar --force para regenerar): {heartwall_path}")

    print(f"Listo. myocyte: {myocyte_dir}, dwt2d: {dwt2d_path}, heartwall: {heartwall_path}")


if __name__ == "__main__":
    main()
