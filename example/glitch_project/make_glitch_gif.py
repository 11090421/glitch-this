"""批量生成故障图片、故障 GIF，并合并为一段长 GIF。"""

import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageSequence

from glitch_this import ImageGlitcher

SRC_DIR = BASE_DIR / "source_imgs"
OUT_DIR = BASE_DIR / "output"
FRAMES_DIR = BASE_DIR / "glitched_frames"
MERGED_DIR = BASE_DIR / "成品"

GIF_ORDER = [
    "glitched_img_01.gif",
    "glitched_img_02.gif",
    "glitched_img-03.gif",
    "glitched_img_04.gif",
    "glitched_img_05.gif",
]

GLITCH_LEVEL = 2.0
COLOR_OFFSET = True
SCAN_LINES = False
GIF_FRAMES = 5
GIF_DURATION = 200  # 每帧时长，单位毫秒
GIF_LOOP = 0
CANVAS_SIZE = (640, 480)
TOTAL_DURATION_LIMIT_MS = 6000
GARBLE_NOISE = True
GARBLE_DENSITY = 0.02
GARBLE_STRENGTH = 25
BLUR_BOUNDARY_SEGMENTS = {2, 3, 4, 5}
BLEND_FRAMES = 2
BLEND_BLUR_RADIUS = 2
SHARPEN_GIF = True
SHARPEN_RADIUS = 1.0
SHARPEN_PERCENT = 100
SHARPEN_THRESHOLD = 3
BLOCK_GLITCH = True
BLOCK_MAX_COUNT = 4
BLOCK_MIN_RATIO = 0.02
BLOCK_MAX_RATIO = 0.12
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def add_garble_noise(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    height, width = arr.shape[:2]
    mask = np.random.rand(height, width) < GARBLE_DENSITY
    noise = np.random.randint(
        -GARBLE_STRENGTH, GARBLE_STRENGTH + 1, size=(height, width, 1)
    )
    result = arr.astype(np.int16)
    result[mask, :3] = np.clip(result[mask, :3] + noise[mask], 0, 255)
    return Image.fromarray(result.astype(np.uint8), img.mode)


def sharpen_frame(img: Image.Image) -> Image.Image:
    return img.filter(
        ImageFilter.UnsharpMask(
            radius=SHARPEN_RADIUS,
            percent=SHARPEN_PERCENT,
            threshold=SHARPEN_THRESHOLD,
        )
    )


def add_block_glitch(img: Image.Image) -> Image.Image:
    result = img.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    for _ in range(random.randint(0, BLOCK_MAX_COUNT)):
        block_w = random.randint(int(width * BLOCK_MIN_RATIO), int(width * BLOCK_MAX_RATIO))
        block_h = random.randint(int(height * BLOCK_MIN_RATIO), int(height * BLOCK_MAX_RATIO))
        x = random.randint(0, width - block_w)
        y = random.randint(0, height - block_h)
        color = tuple(random.randint(0, 255) for _ in range(3))
        if result.mode == "RGBA":
            color += (255,)
        draw.rectangle([x, y, x + block_w, y + block_h], fill=color)
    return result


def make_glitch_images(glitcher: ImageGlitcher) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for src in sorted(SRC_DIR.iterdir()):
        if not src.is_file() or src.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        glitched = glitcher.glitch_image(
            str(src),
            GLITCH_LEVEL,
            color_offset=COLOR_OFFSET,
            scan_lines=SCAN_LINES,
        )
        name = src.name.split(".")[0]
        dst = OUT_DIR / f"glitched_{name}.png"
        glitched.save(dst)
        print(f"Image saved: {dst}")


def make_glitch_gifs(glitcher: ImageGlitcher) -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    for src in sorted(OUT_DIR.iterdir()):
        if not src.is_file() or src.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        frames = glitcher.glitch_image(
            str(src),
            GLITCH_LEVEL,
            gif=True,
            frames=GIF_FRAMES,
            color_offset=COLOR_OFFSET,
            scan_lines=SCAN_LINES,
        )
        if GARBLE_NOISE:
            frames = [add_garble_noise(frame) for frame in frames]
        if BLOCK_GLITCH:
            frames = [add_block_glitch(frame) for frame in frames]
        dst = FRAMES_DIR / f"{src.stem}.gif"
        frames[0].save(
            dst,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=GIF_DURATION,
            loop=GIF_LOOP,
        )
        print(f"GIF saved: {dst}")


def merge_glitch_gifs() -> None:
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    segments = []
    for name in GIF_ORDER:
        segment = []
        with Image.open(FRAMES_DIR / name) as gif:
            for frame in ImageSequence.Iterator(gif):
                img = frame.convert("RGB")
                img.thumbnail(CANVAS_SIZE, Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", CANVAS_SIZE, (0, 0, 0))
                canvas.paste(
                    img,
                    ((CANVAS_SIZE[0] - img.width) // 2, (CANVAS_SIZE[1] - img.height) // 2),
                )
                segment.append((canvas, frame.info.get("duration", GIF_DURATION)))
        segments.append(segment)

    sequence = [item for segment in segments for item in segment]

    for segment_no in sorted(BLUR_BOUNDARY_SEGMENTS, reverse=True):
        boundary = segment_no - 1
        if not 0 < boundary < len(segments):
            continue
        start = sum(len(segments[i]) for i in range(boundary))
        count = min(BLEND_FRAMES, len(segments[boundary - 1]), len(segments[boundary]))
        if count == 0:
            continue
        left = sequence[start - count:start]
        right = sequence[start:start + count]
        blended = []
        for i in range(count):
            alpha = (i + 1) / (count + 1)
            img = Image.blend(left[i][0], right[i][0], alpha)
            img = img.filter(ImageFilter.GaussianBlur(BLEND_BLUR_RADIUS))
            duration = (left[i][1] + right[i][1]) // 2
            blended.append((img, duration))
        sequence[start - count:start + count] = blended

    total_ms = sum(duration for _, duration in sequence)
    if total_ms > TOTAL_DURATION_LIMIT_MS:
        raise ValueError(f"GIF 总时长 {total_ms}ms 超过限制 {TOTAL_DURATION_LIMIT_MS}ms")

    canvas_frames = [sharpen_frame(item[0]) if SHARPEN_GIF else item[0] for item in sequence]
    durations = [item[1] for item in sequence]
    dst = MERGED_DIR / "glitched_all.gif"
    canvas_frames[0].save(
        dst,
        format="GIF",
        append_images=canvas_frames[1:],
        save_all=True,
        duration=durations,
        loop=GIF_LOOP,
    )
    print(f"Merged GIF saved: {dst} ({len(canvas_frames)} frames, {total_ms}ms)")


def main() -> None:
    glitcher = ImageGlitcher()
    make_glitch_images(glitcher)
    make_glitch_gifs(glitcher)
    merge_glitch_gifs()


if __name__ == "__main__":
    main()
