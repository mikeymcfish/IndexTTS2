import html
import importlib.util
import json
import math
import os
import sys
import threading
import time
import glob
import platform
import subprocess
import tempfile
import shutil
import re

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def console_log(message):
    """Emit console logs immediately so users can see progress."""
    print(message, flush=True)

# pandas removed - not needed, using native list format instead

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=80, help="GUI: Max tokens per generation segment")
parser.add_argument("--share", action="store_true", default=False, help="Enable Gradio live sharing to create a public link")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto
from tools.vocal_isolation import (
    VocalIsolationError,
    isolate_vocals_with_demucs,
)

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )
# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = ["Same as speaker voice",
                "Use emotion reference audio",
                "Use emotion vector control",
                "Use emotion text description"]

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)
os.makedirs("outputs/used_audios",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
TEXT_FILE_SIZE_LIMIT = 2 * 1024 * 1024  # 2 MB limit for uploaded text files

# Console logging helpers for long-running segmentation previews
def log_segmentation_progress(prefix, processed, total, start_time, last_log_time=None, min_interval=1.0):
    """Log incremental segmentation progress with ETA estimation."""
    if total <= 0:
        return last_log_time

    now = time.perf_counter()
    if processed == total or last_log_time is None or (now - last_log_time) >= min_interval:
        elapsed = max(now - start_time, 1e-6)
        rate = processed / elapsed
        remaining = total - processed
        if remaining <= 0:
            eta_str = "0.0s"
        elif rate > 0:
            eta = remaining / rate
            eta_str = f"{eta:.1f}s"
        else:
            eta_str = "estimating..."

        console_log(
            f"[{prefix}] {processed}/{total} segments processed "
            f"({elapsed:.1f}s elapsed, ETA {eta_str})"
        )
        return now

    return last_log_time


def log_segmentation_summary(prefix, total_segments, total_tokens, elapsed):
    """Log a summary line for segmentation work."""
    elapsed = max(elapsed, 1e-6)
    if total_segments:
        rate = total_segments / elapsed
        console_log(
            f"[{prefix}] Completed {total_segments} segments "
            f"({total_tokens} tokens) in {elapsed:.2f}s ({rate:.2f} seg/s)"
        )
    else:
        console_log(f"[{prefix}] No segments produced (elapsed {elapsed:.2f}s)")


# Try to import pydub for MP3 export
try:
    from pydub import AudioSegment
    MP3_AVAILABLE = True
except ImportError:
    MP3_AVAILABLE = False
    print("Warning: pydub not installed. MP3 export will not be available.")
    print("To enable MP3 export, install pydub: pip install pydub")

try:
    from mutagen.id3 import ID3, CHAP, CTOC, TIT2
    from mutagen.mp3 import MP3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    print("Warning: mutagen not installed. MP3 chapter metadata will not be added.")


DEMUCS_AVAILABLE = importlib.util.find_spec("demucs") is not None
if not DEMUCS_AVAILABLE:
    print(
        "Warning: demucs not installed. Vocal isolation will be disabled until you run "
        "`pip install demucs`."
    )

# Check if FFmpeg is available
def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

FFMPEG_AVAILABLE = check_ffmpeg()
if not FFMPEG_AVAILABLE:
    print("Warning: FFmpeg not found in PATH. Video/audio processing will not work.")
    print("Please install FFmpeg: https://ffmpeg.org/download.html")

def get_next_file_number(output_dir="outputs", target_folder=None, prefix=""):
    """Get the next available file number in sequence."""
    if target_folder:
        output_dir = target_folder

    os.makedirs(output_dir, exist_ok=True)

    # Find all existing files with our naming pattern
    existing_files = glob.glob(os.path.join(output_dir, f"{prefix}[0-9][0-9][0-9][0-9].*"))

    if not existing_files:
        return 1

    # Extract numbers from filenames
    numbers = []
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        # Remove prefix if present
        if prefix:
            filename = filename[len(prefix):]
        # Extract the 4-digit number
        try:
            num_str = filename[:4]
            if num_str.isdigit():
                numbers.append(int(num_str))
        except:
            continue

    if numbers:
        return max(numbers) + 1
    else:
        return 1

def open_outputs_folder():
    """Open the outputs folder in the system's file manager (cross-platform)."""
    output_dir = os.path.abspath("outputs")
    os.makedirs(output_dir, exist_ok=True)

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(output_dir)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", output_dir])
        else:  # Linux and other Unix-like systems
            subprocess.run(["xdg-open", output_dir])
        print(f"Opened outputs folder: {output_dir}")
    except Exception as e:
        print(f"Failed to open outputs folder: {str(e)}")

def generate_output_path(target_folder=None, filename=None, save_as_mp3=False, prefix=""):
    """Generate output file path with sequential numbering."""
    output_dir = target_folder if target_folder else "outputs"
    os.makedirs(output_dir, exist_ok=True)

    if filename:
        # Use provided filename
        extension = ".mp3" if save_as_mp3 and MP3_AVAILABLE else ".wav"
        if not filename.endswith(('.wav', '.mp3')):
            filename = filename + extension
        return os.path.join(output_dir, filename)
    else:
        # Use sequential numbering
        next_num = get_next_file_number(output_dir, target_folder, prefix)
        extension = ".mp3" if save_as_mp3 and MP3_AVAILABLE else ".wav"
        filename = f"{prefix}{next_num:04d}{extension}"
        return os.path.join(output_dir, filename)

def convert_wav_to_mp3(wav_path, mp3_path, bitrate="256k"):
    """Convert WAV file to MP3 using pydub."""
    if not MP3_AVAILABLE:
        print("Warning: MP3 conversion not available. Keeping WAV format.")
        return wav_path

    try:
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate=bitrate)
        # Remove the original WAV file
        os.remove(wav_path)
        return mp3_path
    except Exception as e:
        print(f"Error converting to MP3: {e}")
        return wav_path


def resolve_uploaded_file_path(file_input):
    """Resolve the filesystem path from a Gradio file input."""
    if not file_input:
        return None

    if isinstance(file_input, str):
        return file_input

    if isinstance(file_input, dict):
        return file_input.get("name") or file_input.get("path")

    return getattr(file_input, "name", None)


def format_timecode(milliseconds):
    """Format milliseconds into HH:MM:SS or MM:SS."""
    try:
        total_ms = max(0, int(milliseconds))
    except (TypeError, ValueError):
        total_ms = 0
    seconds, _ = divmod(total_ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def parse_max_tokens(max_text_tokens_per_segment):
    """Parse and clamp the max tokens per segment value from UI input."""
    if not max_text_tokens_per_segment:
        return 120
    try:
        max_tokens = int(float(str(max_text_tokens_per_segment).strip()))
        return max(20, min(max_tokens, tts.cfg.gpt.max_text_tokens))
    except (ValueError, TypeError):
        return 120


def build_segments_preview_data(text, max_tokens, log_prefix="Segmentation preview (standard)"):
    """Build preview rows for segmented text."""
    if not text or not text.strip():
        return []

    try:
        start_time = time.perf_counter()
        char_count = len(text)
        console_log(f"[{log_prefix}] Tokenizing {char_count} characters...")
        token_start = time.perf_counter()
        text_tokens_list = tts.tokenizer.tokenize(text)
        token_elapsed = time.perf_counter() - token_start
        token_count = len(text_tokens_list)
        console_log(
            f"[{log_prefix}] Tokenized {token_count} tokens in {token_elapsed:.2f}s; splitting into segments (max {max_tokens})."
        )

        segment_start = time.perf_counter()
        segments = tts.tokenizer.split_segments(
            text_tokens_list,
            max_text_tokens_per_segment=max_tokens,
        )
        split_elapsed = time.perf_counter() - segment_start
        console_log(
            f"[{log_prefix}] Created {len(segments)} raw segments in {split_elapsed:.2f}s; assembling preview rows..."
        )

        data = []
        total_tokens = 0
        assembly_start = time.perf_counter()
        last_log_time = None
        for i, segment_tokens in enumerate(segments):
            segment_str = ''.join(segment_tokens)
            tokens_count = len(segment_tokens)
            total_tokens += tokens_count
            data.append([i, segment_str, tokens_count])
            last_log_time = log_segmentation_progress(
                f"{log_prefix} assembly",
                i + 1,
                len(segments),
                assembly_start,
                last_log_time,
                min_interval=0.5,
            )
        elapsed = time.perf_counter() - start_time
        log_segmentation_summary(log_prefix, len(segments), total_tokens, elapsed)
        return data
    except Exception as exc:
        console_log(f"Error building segments preview: {exc}")
        return []


def build_chapter_signature(chapters_state):
    """Create a stable signature from chapter matches for change detection."""
    signature_entries = []
    if isinstance(chapters_state, (list, tuple)):
        for idx, chapter in enumerate(chapters_state):
            if not isinstance(chapter, dict):
                continue
            title = chapter.get("title")
            if isinstance(title, str):
                title = title.strip()
            else:
                title = ""
            if not title:
                title = f"Chapter {idx + 1}"
            try:
                start_char = int(chapter.get("start_char", 0))
            except (TypeError, ValueError):
                start_char = 0
            signature_entries.append((max(0, start_char), title))

    signature_entries.sort(key=lambda item: item[0])
    deduped = []
    for start, title in signature_entries:
        if deduped and start == deduped[-1][0] and title == deduped[-1][1]:
            continue
        deduped.append((start, title))
    return tuple(deduped)


def build_segmentation_chapters(text, chapters_state):
    """Build ordered chapter chunks for chapter-aware segmentation."""
    sanitized_text = text or ""
    total_length = len(sanitized_text)
    if total_length == 0:
        return []

    signature_entries = list(build_chapter_signature(chapters_state))
    chunks = []

    if not signature_entries:
        if sanitized_text.strip():
            chunks.append({
                "title": "Full Text",
                "start": 0,
                "end": total_length,
                "text": sanitized_text,
            })
        return chunks

    # Preface chunk before first chapter
    first_start = signature_entries[0][0]
    if first_start > 0:
        prefix_text = sanitized_text[:first_start]
        if prefix_text.strip():
            chunks.append({
                "title": "Preface",
                "start": 0,
                "end": first_start,
                "text": prefix_text,
            })

    for entry_idx, (start_char, title) in enumerate(signature_entries):
        start = max(0, min(total_length, start_char))
        next_start = total_length
        if entry_idx + 1 < len(signature_entries):
            next_start = max(start, min(total_length, signature_entries[entry_idx + 1][0]))
        chapter_text = sanitized_text[start:next_start]
        if not chapter_text.strip():
            continue
        chunks.append({
            "title": title,
            "start": start,
            "end": next_start,
            "text": chapter_text,
        })

    if chunks:
        last_end = chunks[-1]["end"]
    else:
        last_end = 0

    if last_end < total_length:
        trailing_text = sanitized_text[last_end:]
        if trailing_text.strip():
            chunks.append({
                "title": "Postscript",
                "start": last_end,
                "end": total_length,
                "text": trailing_text,
            })

    # Ensure index order metadata
    for idx, chunk in enumerate(chunks):
        chunk["index"] = idx

    return chunks


def format_token_analysis(total_tokens, segment_count, max_tokens, chapter_count=None):
    """Format a human-readable token analysis message."""
    if total_tokens is None or segment_count is None:
        return ""

    segment_part = f"across {segment_count} segment{'s' if segment_count != 1 else ''}" if segment_count else "with no segments"
    message = (
        f"📊 Token analysis: {total_tokens} token{'s' if total_tokens != 1 else ''} "
        f"{segment_part} (max {max_tokens} tokens/segment)."
    )

    if chapter_count is not None:
        message += f" Chapter segmentation active for {chapter_count} chapter{'s' if chapter_count != 1 else ''}."

    return message


def build_chapter_table(chapters, text_length, total_duration_ms=None):
    """Create table rows for chapter preview display."""
    rows = []
    safe_text_length = max(1, int(text_length) if text_length else 1)
    for idx, chapter in enumerate(chapters or []):
        title = chapter.get("title") or f"Chapter {idx + 1}"
        start_char = max(0, int(chapter.get("start_char", 0)))

        if "start_ms" in chapter:
            start_display = format_timecode(chapter.get("start_ms", 0))
        elif total_duration_ms is not None:
            ratio = min(1.0, start_char / safe_text_length)
            start_display = format_timecode(int(total_duration_ms * ratio))
        else:
            ratio = start_char / safe_text_length
            start_display = f"{ratio * 100:.1f}% of text"

        rows.append([idx + 1, title, start_char, start_display])
    return rows


def apply_chapters_to_mp3(mp3_path, text, chapters):
    """Apply chapter metadata to an MP3 file using mutagen."""
    if not MUTAGEN_AVAILABLE:
        raise RuntimeError("mutagen is not available to write chapter metadata")
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"MP3 file not found: {mp3_path}")

    audio = MP3(mp3_path, ID3=ID3)
    total_duration_ms = int(audio.info.length * 1000) if audio.info and audio.info.length else 0
    text_length = len(text or "")

    sorted_chapters = sorted(
        [
            {
                "title": (chapter.get("title") or f"Chapter {idx + 1}").strip() or f"Chapter {idx + 1}",
                "start_char": max(0, int(chapter.get("start_char", 0))),
            }
            for idx, chapter in enumerate(chapters or [])
        ],
        key=lambda item: item["start_char"],
    )

    if not sorted_chapters:
        return total_duration_ms, []

    if audio.tags is None:
        audio.add_tags()
    else:
        audio.tags.delall("CHAP")
        audio.tags.delall("CTOC")

    safe_text_length = max(1, text_length)
    chapter_entries = []
    for chapter in sorted_chapters:
        ratio = min(1.0, chapter["start_char"] / safe_text_length)
        start_ms = int(total_duration_ms * ratio) if total_duration_ms else 0
        chapter_entries.append(
            {
                "title": chapter["title"],
                "start_char": chapter["start_char"],
                "start_ms": start_ms,
            }
        )

    # Ensure start times are non-decreasing
    for i in range(1, len(chapter_entries)):
        if chapter_entries[i]["start_ms"] < chapter_entries[i - 1]["start_ms"]:
            chapter_entries[i]["start_ms"] = chapter_entries[i - 1]["start_ms"]

    element_ids = []
    for idx, chapter in enumerate(chapter_entries):
        element_id = f"chp{idx:04d}"
        start_ms = chapter["start_ms"]
        if idx + 1 < len(chapter_entries):
            end_ms = chapter_entries[idx + 1]["start_ms"]
        else:
            end_ms = total_duration_ms or start_ms
        chap_frame = CHAP(
            element_id=element_id,
            start_time=start_ms,
            end_time=end_ms,
            start_offset=0,
            end_offset=0,
            sub_frames=[TIT2(encoding=3, text=chapter["title"])],
        )
        audio.tags.add(chap_frame)
        element_ids.append(element_id)

    if element_ids:
        toc = CTOC(
            element_id="toc",
            flags=0,
            child_element_ids=element_ids,
        )
        toc.add(TIT2(encoding=3, text="Chapters"))
        audio.tags.add(toc)

    audio.save()
    return total_duration_ms, chapter_entries


def read_mp3_chapters(mp3_path):
    """Read existing chapter frames from an MP3 file."""
    if not MUTAGEN_AVAILABLE or not os.path.exists(mp3_path):
        return []

    try:
        audio = MP3(mp3_path, ID3=ID3)
    except Exception as exc:
        console_log(f"Failed to inspect chapters in {mp3_path}: {exc}")
        return []

    if not audio.tags:
        return []

    chapters = []
    for key, frame in audio.tags.items():
        if not key.startswith("CHAP"):
            continue

        start_time = int(getattr(frame, "start_time", 0) or 0)
        title = None

        for attr in ("subframes", "sub_frames"):
            subframes = getattr(frame, attr, None)
            if subframes is None:
                continue

            try:
                tit2_frames = subframes.getall("TIT2")
            except AttributeError:
                tit2_candidate = subframes.get("TIT2") if hasattr(subframes, "get") else None
                tit2_frames = [tit2_candidate] if tit2_candidate else []

            for tit2 in tit2_frames or []:
                if tit2 is None:
                    continue
                text = getattr(tit2, "text", [])
                if isinstance(text, (list, tuple)):
                    title = next((str(item) for item in text if item), None)
                elif text:
                    title = str(text)
                if title:
                    break
            if title:
                break

        if not title:
            try:
                for tit2 in frame.getall("TIT2"):
                    text = getattr(tit2, "text", [])
                    if isinstance(text, (list, tuple)):
                        title = next((str(item) for item in text if item), None)
                    elif text:
                        title = str(text)
                    if title:
                        break
            except Exception:
                title = None

        if not title:
            title = os.path.splitext(os.path.basename(mp3_path))[0] or "Chapter"

        chapters.append({
            "title": title,
            "start_ms": max(0, start_time),
        })

    return sorted(chapters, key=lambda item: item["start_ms"])


def write_chapters_to_mp3(mp3_path, chapter_entries):
    """Write chapter metadata to an MP3 file."""
    if not MUTAGEN_AVAILABLE:
        return False, "mutagen is not installed; cannot write chapters."
    if not os.path.exists(mp3_path):
        return False, f"MP3 file not found: {mp3_path}"

    audio = MP3(mp3_path, ID3=ID3)
    total_duration_ms = int(audio.info.length * 1000) if audio.info and audio.info.length else 0

    if audio.tags is None:
        audio.add_tags()
    else:
        audio.tags.delall("CHAP")
        audio.tags.delall("CTOC")

    cleaned = []
    for entry in chapter_entries or []:
        if entry is None:
            continue
        title = str(entry.get("title") or "Chapter").strip() or "Chapter"
        start_ms = int(max(0, entry.get("start_ms", 0)))
        cleaned.append({"title": title, "start_ms": start_ms})

    if not cleaned:
        audio.save()
        return True, "No chapters to write."

    cleaned.sort(key=lambda item: item["start_ms"])

    if total_duration_ms <= 0 and cleaned:
        total_duration_ms = cleaned[-1]["start_ms"]

    element_ids = []
    for idx, chapter in enumerate(cleaned):
        element_id = f"chp{idx:04d}"
        start_ms = chapter["start_ms"]
        if idx + 1 < len(cleaned):
            end_ms = max(start_ms, cleaned[idx + 1]["start_ms"])
        else:
            end_ms = max(start_ms, total_duration_ms)

        chap_frame = CHAP(
            element_id=element_id,
            start_time=start_ms,
            end_time=end_ms,
            start_offset=0,
            end_offset=0,
            sub_frames=[TIT2(encoding=3, text=chapter["title"])],
        )
        audio.tags.add(chap_frame)
        element_ids.append(element_id)

    if element_ids:
        toc = CTOC(
            element_id="toc",
            flags=0,
            child_element_ids=element_ids,
        )
        toc.add(TIT2(encoding=3, text="Chapters"))
        audio.tags.add(toc)

    audio.save()
    return True, f"Wrote {len(element_ids)} chapter marker(s)."


def merge_mp3_files(mp3_files, keep_existing_chapters=True, output_filename="", mp3_bitrate_value="256k"):
    """Merge multiple MP3 files into a single MP3 and manage chapter metadata."""
    if not MP3_AVAILABLE:
        return (
            gr.update(value="MP3 merging requires pydub. Install pydub to enable this feature.", visible=True),
            gr.update(value=None, visible=False),
        )

    file_paths = []
    if isinstance(mp3_files, list):
        for item in mp3_files:
            if isinstance(item, str) and item:
                file_paths.append(item)
            elif isinstance(item, dict):
                path = item.get("name") or item.get("path")
                if path:
                    file_paths.append(path)
    elif isinstance(mp3_files, str) and mp3_files:
        file_paths.append(mp3_files)

    file_paths = [path for path in file_paths if path and os.path.exists(path)]

    if len(file_paths) < 2:
        return (
            gr.update(value="Please select at least two MP3 files to merge.", visible=True),
            gr.update(value=None, visible=False),
        )

    bitrate = str(mp3_bitrate_value or "256k")
    if not bitrate.endswith("k"):
        bitrate = f"{bitrate}k"

    combined_audio = None
    chapter_entries = []
    offset_ms = 0
    unable_to_keep = False

    for idx, path in enumerate(file_paths):
        try:
            segment = AudioSegment.from_file(path)
        except Exception as exc:
            return (
                gr.update(
                    value=f"Failed to load MP3 '{os.path.basename(path)}': {exc}",
                    visible=True,
                ),
                gr.update(value=None, visible=False),
            )

        if combined_audio is None:
            combined_audio = segment
        else:
            combined_audio += segment

        base_title = os.path.splitext(os.path.basename(path))[0] or f"Track {idx + 1}"
        chapter_entries.append({
            "title": base_title,
            "start_ms": offset_ms,
        })

        if keep_existing_chapters:
            if MUTAGEN_AVAILABLE:
                for chapter in read_mp3_chapters(path):
                    chapter_entries.append({
                        "title": chapter.get("title") or base_title,
                        "start_ms": offset_ms + int(chapter.get("start_ms", 0)),
                    })
            else:
                unable_to_keep = True

        offset_ms += len(segment)

    if combined_audio is None:
        return (
            gr.update(value="No audio data could be read from the selected files.", visible=True),
            gr.update(value=None, visible=False),
        )

    output_path = generate_output_path(
        filename=output_filename.strip() or None,
        save_as_mp3=True,
        prefix="merged_",
    )

    try:
        combined_audio.export(output_path, format="mp3", bitrate=bitrate)
    except Exception as exc:
        return (
            gr.update(value=f"Failed to export merged MP3: {exc}", visible=True),
            gr.update(value=None, visible=False),
        )

    messages = [
        f"Merged {len(file_paths)} MP3 file(s) into {os.path.relpath(output_path)}.",
    ]

    if chapter_entries:
        if MUTAGEN_AVAILABLE:
            success, chapter_msg = write_chapters_to_mp3(output_path, chapter_entries)
            messages.append(chapter_msg)
        else:
            messages.append("mutagen is not installed; chapter markers were not written.")

    if unable_to_keep:
        messages.append("Existing chapters could not be preserved because mutagen is missing.")

    return (
        gr.update(value="\n".join(messages), visible=True),
        gr.update(value=output_path, visible=True),
    )


def isolate_vocals_ui(
    audio_file,
    model_name,
    device_choice,
    segment_length,
    shifts,
    overlap,
    export_mp3,
    mp3_bitrate_value,
    progress=gr.Progress(track_tqdm=False),
):
    """Gradio callback to isolate vocals via Demucs."""

    if not DEMUCS_AVAILABLE:
        return (
            gr.update(
                value="Demucs is not installed. Install it with `pip install demucs` to enable vocal isolation.",
                visible=True,
            ),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        )

    resolved_path = resolve_uploaded_file_path(audio_file)

    if not resolved_path:
        return (
            gr.update(value="Please upload an audio file to process.", visible=True),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        )

    if not os.path.exists(resolved_path):
        return (
            gr.update(value=f"File not found: {resolved_path}", visible=True),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        )

    device = None if device_choice in (None, "", "Auto") else device_choice
    segment = segment_length if segment_length and segment_length > 0 else None
    shifts_value = int(shifts) if shifts and shifts > 0 else None
    overlap_value = overlap if overlap is not None else None

    try:
        progress(0.0, desc="Starting Demucs")
        result = isolate_vocals_with_demucs(
            resolved_path,
            model_name=model_name,
            device=device,
            segment_length=segment,
            shifts=shifts_value,
            overlap=overlap_value,
        )
    except ModuleNotFoundError as exc:
        return (
            gr.update(value=str(exc), visible=True),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        )
    except (VocalIsolationError, FileNotFoundError) as exc:
        return (
            gr.update(value=str(exc), visible=True),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        )
    except Exception as exc:  # pylint: disable=broad-except
        return (
            gr.update(value=f"Unexpected error: {exc}", visible=True),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        )

    progress(0.6, desc="Finalizing stems")

    mp3_requested = bool(export_mp3)
    bitrate = str(mp3_bitrate_value or "256k")
    if mp3_requested and not bitrate.endswith("k"):
        bitrate = f"{bitrate}k"

    vocals_output_path = None
    instrumental_output_path = None
    messages = ["✅ Vocal isolation complete."]

    try:
        if mp3_requested and MP3_AVAILABLE:
            vocals_output_path = generate_output_path(prefix="vocals_", save_as_mp3=True)
            convert_wav_to_mp3(result.vocals_path, vocals_output_path, bitrate=bitrate)
        else:
            if mp3_requested and not MP3_AVAILABLE:
                messages.append("⚠️ pydub is not installed; saved vocals as WAV instead of MP3.")
            vocals_output_path = generate_output_path(prefix="vocals_", save_as_mp3=False)
            shutil.move(result.vocals_path, vocals_output_path)

        if result.accompaniment_path and os.path.exists(result.accompaniment_path):
            instrumental_output_path = generate_output_path(prefix="instrumental_", save_as_mp3=False)
            shutil.move(result.accompaniment_path, instrumental_output_path)
            messages.append(
                f"Instrumental stem saved to `{os.path.relpath(instrumental_output_path)}`."
            )

        messages.append(f"Vocals saved to `{os.path.relpath(vocals_output_path)}`.")
    finally:
        shutil.rmtree(result.workspace_dir, ignore_errors=True)

    progress(1.0, desc="Done")

    logs_visible = bool(result.logs.strip())

    return (
        gr.update(value="\n".join(messages), visible=True),
        gr.update(value=vocals_output_path, visible=bool(vocals_output_path)),
        gr.update(value=instrumental_output_path, visible=bool(instrumental_output_path)),
        gr.update(value=result.logs, visible=logs_visible),
    )


def load_mp3_chapter_data(mp3_file):
    """Load chapter metadata from an MP3 for editing."""
    resolved_path = resolve_uploaded_file_path(mp3_file)

    if not resolved_path:
        return (
            gr.update(value=[], visible=False),
            gr.update(value="Select an MP3 file to inspect its chapters.", visible=True),
            {"path": "", "chapters": []},
            gr.update(interactive=False),
        )

    if not os.path.exists(resolved_path):
        return (
            gr.update(value=[], visible=False),
            gr.update(value=f"File not found: {resolved_path}", visible=True),
            {"path": "", "chapters": []},
            gr.update(interactive=False),
        )

    if not MUTAGEN_AVAILABLE:
        return (
            gr.update(value=[], visible=False),
            gr.update(value="mutagen is required to read MP3 chapters.", visible=True),
            {"path": "", "chapters": []},
            gr.update(interactive=False),
        )

    chapters = read_mp3_chapters(resolved_path)
    filename = os.path.basename(resolved_path)

    if not chapters:
        return (
            gr.update(value=[], visible=False),
            gr.update(value=f"No chapters were found in {filename}.", visible=True),
            {"path": resolved_path, "chapters": []},
            gr.update(interactive=False),
        )

    table_rows = [[chapter.get("title", ""), format_timecode(chapter.get("start_ms", 0))] for chapter in chapters]
    status_message = f"Loaded {len(chapters)} chapter(s) from {filename}. Edit the Title column and click Save."

    return (
        gr.update(value=table_rows, visible=True),
        gr.update(value=status_message, visible=True),
        {"path": resolved_path, "chapters": chapters},
        gr.update(interactive=True),
    )


def save_mp3_chapter_titles(table_rows, editor_state):
    """Persist edited chapter titles back into the MP3 file."""
    state = editor_state or {}
    resolved_path = state.get("path")
    original_chapters = state.get("chapters") or []

    if not resolved_path:
        return (
            gr.update(value="Load an MP3 file with chapters before saving.", visible=True),
            {"path": "", "chapters": []},
            gr.update(value=[], visible=False),
        )

    if not os.path.exists(resolved_path):
        return (
            gr.update(value=f"File no longer exists: {resolved_path}", visible=True),
            {"path": "", "chapters": []},
            gr.update(value=[], visible=False),
        )

    if not MUTAGEN_AVAILABLE:
        return (
            gr.update(value="mutagen is required to write MP3 chapters.", visible=True),
            {"path": resolved_path, "chapters": original_chapters},
            gr.update(value=table_rows or [], visible=bool(table_rows)),
        )

    if not original_chapters:
        return (
            gr.update(value="No chapters are available to edit for this file.", visible=True),
            {"path": resolved_path, "chapters": []},
            gr.update(value=[], visible=False),
        )

    titles = []
    table_rows = table_rows or []
    for idx, chapter in enumerate(original_chapters):
        base_title = chapter.get("title") or f"Chapter {idx + 1}"
        new_title = base_title
        if idx < len(table_rows):
            row = table_rows[idx]
            if isinstance(row, (list, tuple)) and row:
                candidate = row[0]
            elif isinstance(row, dict):
                candidate = row.get("Title") or row.get(0)
            else:
                candidate = None
            if candidate is not None:
                candidate_str = str(candidate).strip()
                if candidate_str:
                    new_title = candidate_str
        titles.append({"title": new_title, "start_ms": int(chapter.get("start_ms", 0))})

    success, message = write_chapters_to_mp3(resolved_path, titles)

    if success:
        updated_chapters = []
        for idx, chapter in enumerate(original_chapters):
            updated = {
                "title": titles[idx]["title"],
                "start_ms": int(chapter.get("start_ms", 0)),
            }
            updated_chapters.append(updated)
        updated_table = [[entry["title"], format_timecode(entry.get("start_ms", 0))] for entry in updated_chapters]
        return (
            gr.update(value=f"{message} ({os.path.basename(resolved_path)})", visible=True),
            {"path": resolved_path, "chapters": updated_chapters},
            gr.update(value=updated_table, visible=True),
        )

    return (
        gr.update(value=message, visible=True),
        {"path": resolved_path, "chapters": original_chapters},
        gr.update(value=table_rows, visible=bool(table_rows)),
    )


def compute_chapter_preview_data(enable_chapters, text, regex_pattern):
    """Compute chapter preview matches based on a regex pattern."""
    if not enable_chapters:
        return [], "", [], False

    sanitized_text = text or ""
    if not sanitized_text.strip():
        return [], "Enter text to analyze for chapters.", [], True

    if not regex_pattern or not regex_pattern.strip():
        return [], "Enter a regex to detect chapter headings.", [], True

    try:
        pattern = re.compile(regex_pattern, flags=re.MULTILINE)
    except re.error as exc:
        return [], f"Regex error: {exc}", [], True

    matches = []
    for idx, match in enumerate(pattern.finditer(sanitized_text)):
        start_char = match.start()
        title = match.group().strip() or f"Chapter {idx + 1}"
        matches.append({
            "title": title,
            "start_char": start_char,
        })
        if len(matches) >= 200:
            break

    if not matches:
        return [], "No chapters matched the provided regex.", [], True

    table_rows = build_chapter_table(matches, len(sanitized_text))
    return matches, "", table_rows, True

def extract_audio_from_media(media_path, output_path=None, sample_rate=24000):
    """Extract audio from video/audio file and convert to acceptable format using FFmpeg."""
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    try:
        # Use FFmpeg subprocess directly for cross-platform compatibility
        cmd = [
            'ffmpeg', '-i', media_path,
            '-ar', str(sample_rate),
            '-ac', '1',  # mono
            '-acodec', 'pcm_s16le',
            '-f', 'wav',
            output_path,
            '-y',  # overwrite output
            '-loglevel', 'error'  # only show errors
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None

        if os.path.exists(output_path):
            return output_path
        else:
            return None

    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in PATH.")
        return None
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def extract_time_ranges(audio_path, time_ranges_str, sample_rate=24000):
    """Extract and merge audio segments based on time ranges using FFmpeg.
    Time ranges format: '1:3; 3:7; 11:15'
    """
    try:
        # Parse time ranges
        segments = []
        for range_str in time_ranges_str.split(';'):
            range_str = range_str.strip()
            if ':' in range_str:
                parts = range_str.split(':')
                if len(parts) == 2:
                    start, end = parts
                    try:
                        start_sec = float(start.strip())
                        end_sec = float(end.strip())
                        duration = end_sec - start_sec
                        if duration > 0:
                            segments.append((start_sec, duration))
                    except ValueError:
                        print(f"Invalid time range: {range_str}")
                        continue

        if not segments:
            return None

        # Create a temporary directory for segment files
        temp_dir = tempfile.mkdtemp()
        segment_files = []

        try:
            # Extract each segment using FFmpeg
            for i, (start, duration) in enumerate(segments):
                segment_file = os.path.join(temp_dir, f"segment_{i:03d}.wav")

                cmd = [
                    'ffmpeg', '-i', audio_path,
                    '-ss', str(start),  # start time
                    '-t', str(duration),  # duration
                    '-ar', str(sample_rate),
                    '-ac', '1',  # mono
                    '-acodec', 'pcm_s16le',
                    '-f', 'wav',
                    segment_file,
                    '-y',
                    '-loglevel', 'error'
                ]

                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if result.returncode == 0 and os.path.exists(segment_file):
                    segment_files.append(segment_file)
                else:
                    print(f"Failed to extract segment {start}-{start+duration}: {result.stderr}")

            if not segment_files:
                return None

            # Merge all segments using FFmpeg concat
            output_path = tempfile.mktemp(suffix=".wav")

            if len(segment_files) == 1:
                # If only one segment, just copy it
                shutil.copy2(segment_files[0], output_path)
            else:
                # Create a concat file list
                concat_file = os.path.join(temp_dir, "concat_list.txt")
                with open(concat_file, 'w') as f:
                    for seg_file in segment_files:
                        # Use forward slashes for FFmpeg compatibility
                        seg_path = seg_file.replace(os.sep, '/')
                        f.write(f"file '{seg_path}'\n")

                # Concatenate using FFmpeg
                cmd = [
                    'ffmpeg', '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-ar', str(sample_rate),
                    '-ac', '1',
                    '-acodec', 'pcm_s16le',
                    '-f', 'wav',
                    output_path,
                    '-y',
                    '-loglevel', 'error'
                ]

                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if result.returncode != 0:
                    print(f"Failed to merge segments: {result.stderr}")
                    return None

            return output_path if os.path.exists(output_path) else None

        finally:
            # Clean up temporary files
            for seg_file in segment_files:
                if os.path.exists(seg_file):
                    os.remove(seg_file)
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

    except Exception as e:
        print(f"Error extracting time ranges: {e}")
        return None

def load_audio_from_path(audio_path):
    """Load audio from a file path."""
    if os.path.exists(audio_path):
        return audio_path
    else:
        return None

def gen_single(emo_control_method,prompt, text, save_used_audio, output_filename,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment,
               chapter_segments,
               enable_chapters,
               chapters_state,
               save_as_mp3,
               # Expert params (in order from expert_params list)
               diffusion_steps,
               inference_cfg_rate,
               interval_silence,
               max_speaker_audio_length,
               max_emotion_audio_length,
               autoregressive_batch_size,
               apply_emo_bias,
               max_emotion_sum,
               latent_multiplier,
               max_consecutive_silence,
               mp3_bitrate,
               # Advanced params (in order from advanced_params list)
               do_sample,
               top_p,
               top_k,
               temperature,
               length_penalty,
               num_beams,
               repetition_penalty,
               max_mel_tokens,
               low_memory_mode,
               prevent_vram_accumulation,
               # Model params (semantic layer, cache, emotion biases)
               semantic_layer,
               cfm_cache_length,
               emo_bias_joy,
               emo_bias_anger,
               emo_bias_sad,
               emo_bias_fear,
               emo_bias_disgust,
               emo_bias_depression,
               emo_bias_surprise,
               emo_bias_calm,
               progress=gr.Progress()):
    # Generate output path with sequential numbering or use custom filename
    temp_wav_path = generate_output_path(filename=output_filename, save_as_mp3=False)  # Always generate WAV first
    output_path = temp_wav_path
    # set gradio progress
    tts.gr_progress = progress

    # Update the low memory mode setting
    tts.hybrid_model_device = bool(low_memory_mode)

    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        # Use custom emotion biases if provided
        custom_emo_biases = [emo_bias_joy, emo_bias_anger, emo_bias_sad, emo_bias_fear,
                            emo_bias_disgust, emo_bias_depression, emo_bias_surprise, emo_bias_calm]
        vec = tts.normalize_emo_vec(vec, apply_bias=apply_emo_bias, max_emotion_sum=max_emotion_sum,
                                   custom_biases=custom_emo_biases if apply_emo_bias else None)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")

    # Ensure max_text_tokens_per_segment is within valid range
    max_tokens = parse_max_tokens(max_text_tokens_per_segment)

    # Pass new parameters to infer
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=max_tokens,
                       chapter_segments=chapter_segments,
                       interval_silence=int(interval_silence),
                       diffusion_steps=int(diffusion_steps),
                       inference_cfg_rate=float(inference_cfg_rate),
                       max_speaker_audio_length=float(max_speaker_audio_length),
                       max_emotion_audio_length=float(max_emotion_audio_length),
                       autoregressive_batch_size=int(autoregressive_batch_size),
                       max_emotion_sum=float(max_emotion_sum),
                       latent_multiplier=float(latent_multiplier),
                       max_consecutive_silence=int(max_consecutive_silence),
                       semantic_layer=int(semantic_layer),
                       cfm_cache_length=int(cfm_cache_length),
                       reset_beam_cache_per_segment=bool(prevent_vram_accumulation),
                       **kwargs)

    # Save used audio if requested
    if save_used_audio and prompt:
        try:
            # Extract base filename from output path
            base_name = os.path.basename(output).rsplit('.', 1)[0]
            used_audio_path = os.path.join("outputs/used_audios", f"{base_name}_reference.wav")
            shutil.copy2(prompt, used_audio_path)
            print(f"Saved used reference audio to: {used_audio_path}")
        except Exception as e:
            print(f"Error saving used audio: {e}")

    # Prepare chapter status defaults
    chapter_status_update = gr.update(value="", visible=False)
    chapter_preview_update = gr.update()
    updated_chapter_state = list(chapters_state) if isinstance(chapters_state, (list, tuple)) else []

    # Convert to MP3 if requested
    if save_as_mp3 and MP3_AVAILABLE:
        mp3_path = output.replace('.wav', '.mp3')
        output = convert_wav_to_mp3(output, mp3_path, bitrate=mp3_bitrate)

        if enable_chapters:
            if not updated_chapter_state:
                chapter_status_update = gr.update(
                    value="Chapter regex did not match any sections; no chapters added.",
                    visible=True,
                )
                chapter_preview_update = gr.update(value=[], visible=False, type="array")
            elif not MUTAGEN_AVAILABLE:
                chapter_status_update = gr.update(
                    value="mutagen is not available. Install mutagen to embed MP3 chapter metadata.",
                    visible=True,
                )
            else:
                try:
                    total_ms, chapter_entries = apply_chapters_to_mp3(
                        output,
                        text,
                        updated_chapter_state,
                    )
                    if chapter_entries:
                        chapter_status_update = gr.update(
                            value=f"Added {len(chapter_entries)} chapter marker(s) to the MP3.",
                            visible=True,
                        )
                        updated_chapter_state = chapter_entries
                        table_rows = build_chapter_table(
                            updated_chapter_state,
                            len(text or ""),
                            total_duration_ms=total_ms,
                        )
                        chapter_preview_update = gr.update(value=table_rows, visible=True, type="array")
                    else:
                        chapter_status_update = gr.update(
                            value="Chapter regex did not match any sections; no chapters added.",
                            visible=True,
                        )
                        chapter_preview_update = gr.update(value=[], visible=False, type="array")
                except Exception as exc:
                    chapter_status_update = gr.update(
                        value=f"Failed to add chapters: {exc}",
                        visible=True,
                    )
    elif enable_chapters:
        if not MP3_AVAILABLE:
            chapter_status_update = gr.update(
                value="MP3 conversion is unavailable; chapters can only be embedded into MP3 files.",
                visible=True,
            )
        else:
            chapter_status_update = gr.update(
                value="Enable 'Save as MP3' to include chapter metadata in the output.",
                visible=True,
            )

    return (
        gr.update(value=output, visible=True),
        chapter_status_update,
        chapter_preview_update,
        updated_chapter_state,
    )

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button, update_button


theme = gr.themes.Soft()
theme.font = [gr.themes.GoogleFont("Inter"), "Tahoma", "ui-sans-serif", "system-ui", "sans-serif"]
with gr.Blocks(title="SECourses IndexTTS2 Premium App", theme=theme) as demo:
    mutex = threading.Lock()
    gr.Markdown("## SECourses Index TTS2 Premium App V3 : https://www.patreon.com/posts/139297407")

    with gr.Tab("Audio Generation"):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)

            # Left column for reference audio
            with gr.Column():
                # Video/Audio upload panel (initially visible)
                with gr.Group(visible=True) as media_upload_group:
                    media_upload = gr.File(
                        label="Upload Video/Audio File",
                        file_count="single",
                        file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
                                    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus"],
                        type="filepath"
                    )

                    # Time range extraction section
                    with gr.Column():
                        time_ranges_input = gr.Textbox(
                            label="Extract Audio Segments (optional)",
                            placeholder="e.g., 1:3; 3:7; 11:15 (extracts and merges segments from 1-3s, 3-7s, 11-15s)",
                            value="",
                            info="Enter time ranges to extract specific parts of the audio"
                        )
                        extract_button = gr.Button("Extract and Use Audio", variant="secondary")

                # Speaker reference audio (initially hidden, shown after upload)
                prompt_audio = gr.Audio(
                    label="Speaker Reference Audio (3-90 seconds)",
                    key="prompt_audio",
                    sources=["upload","microphone"],
                    type="filepath",
                    visible=False
                )

                prompt_list = os.listdir("prompts")
                default = ''
                if prompt_list:
                    default = prompt_list[0]

            # Middle column for text input
            with gr.Column():
                text_file_upload = gr.File(
                    label="Upload Text File", file_count="single", file_types=[".txt"], type="filepath"
                )
                text_file_status = gr.Markdown(value="", visible=False)
                token_analysis = gr.Markdown(value="", visible=False)
                input_text_single = gr.TextArea(
                    label="Text to Synthesize",
                    key="input_text_single",
                    placeholder="Enter the text you want to convert to speech",
                    info=f"Model v{tts.model_version or '1.0'} | Supports multiple languages. Long texts will be automatically segmented."
                )
                with gr.Row():
                    process_segments_button = gr.Button("Process Segments", variant="secondary")
                    process_and_generate_button = gr.Button("Process & Generate Audio", variant="primary")
                    generate_from_processed_button = gr.Button("Generate from Processed", variant="secondary")
                    open_outputs_button = gr.Button("📁 Open Outputs Folder", key="open_outputs_button")

                # Output filename and save used audio options
                with gr.Row():
                    output_filename = gr.Textbox(
                        label="Output Filename (optional)",
                        placeholder="Leave empty for auto-numbering (e.g., 0001.wav)",
                        value=""
                    )
                    save_used_audio = gr.Checkbox(
                        label="Save Used Reference Audio",
                        value=False,
                        info="Save reference audio to outputs/used_audios"
                    )

            # Right column for output and load from path
            with gr.Column():
                output_audio = gr.Audio(
                    label="Generated Result (click to play/download)",
                    visible=True,
                    key="output_audio"
                )

                # Load audio from path section
                with gr.Group():
                    gr.Markdown("### Load Audio from Path")
                    with gr.Row():
                        audio_path_input = gr.Textbox(
                            label="Audio File Path",
                            placeholder="Enter full path to audio file",
                            value=""
                        )
                        load_audio_button = gr.Button("Load Audio", variant="secondary")
                    load_status = gr.Textbox(
                        label="Status",
                        value="",
                        interactive=False,
                        visible=False
                    )

        with gr.Accordion("Chapter Settings", open=False):
            enable_chapters = gr.Checkbox(
                label="Enable MP3 Chapters",
                value=False,
                info="When enabled, matches in the regex below will become chapter markers embedded into the exported MP3.",
            )
            chapter_regex_input = gr.Textbox(
                label="Chapter Start Regex",
                value=r"^Chapter\s+\d+",
                placeholder=r"e.g., ^Chapter \d+",
                info="Provide a regular expression that matches the beginning of each chapter. Use multiline mode constructs such as ^ for line starts.",
            )
            chapter_error = gr.Markdown(value="", visible=False)
            chapter_preview = gr.Dataframe(
                headers=["Chapter #", "Title", "Start Char", "Preview Start"],
                interactive=False,
                visible=False,
                type="array",
            )
            chapter_status = gr.Markdown(value="", visible=False)

        with gr.Accordion("Function Settings"):
            # 情感控制选项部分 - now showing ALL options including experimental
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0],
                    label="Emotion Control Method",
                    info="Choose how to control emotions: Speaker's natural emotion, reference audio emotion, manual vector control, or text description"
                )
        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(
                    label="Upload Emotion Reference Audio",
                    type="filepath"
                )

        # 情感随机采样
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(
                label="Random Emotion Sampling",
                value=False,
                info="Enable random sampling from emotion matrix for more varied emotional expression"
            )

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label="Joy", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Happiness and cheerfulness in voice")
                    vec2 = gr.Slider(label="Anger", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Aggressive and forceful tone")
                    vec3 = gr.Slider(label="Sadness", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Melancholic and sorrowful expression")
                    vec4 = gr.Slider(label="Fear", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Anxious and worried tone")
                with gr.Column():
                    vec5 = gr.Slider(label="Disgust", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Repulsed and disgusted expression")
                    vec6 = gr.Slider(label="Depression", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Low energy and melancholic mood")
                    vec7 = gr.Slider(label="Surprise", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Shocked and amazed reaction")
                    vec8 = gr.Slider(label="Calm", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Neutral and peaceful tone")

        with gr.Group(visible=False) as emo_text_group:
            with gr.Row():
                emo_text = gr.Textbox(label="Emotion Description Text",
                                      placeholder="Enter emotion description (or leave empty to automatically use target text as emotion description)",
                                      value="",
                                      info="e.g.: feeling wronged, danger is approaching quietly")

        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(
                label="Emotion Weight",
                minimum=0.0,
                maximum=1.0,
                value=0.65,
                step=0.01,
                info="Controls the strength of emotion blending. 0 = no emotion, 1 = full emotion from reference. Default: 0.65"
            )

        with gr.Accordion("Advanced Generation Parameter Settings", open=True, visible=True) as advanced_settings_group:
            # Row 1: Diffusion Steps and CFG Rate
            with gr.Row():
                diffusion_steps = gr.Slider(
                    label="Diffusion Steps",
                    value=25,
                    minimum=10,
                    maximum=100,
                    step=1,
                    info="Number of denoising steps in the diffusion model. Higher = better quality but slower. Default: 25"
                )
                inference_cfg_rate = gr.Slider(
                    label="CFG Rate (Classifier-Free Guidance)",
                    value=0.7,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    info="Controls how strongly the model follows the voice, emotion, and style characteristics from reference audio. Higher values = stricter adherence to reference, lower = more variation. 0.0 = no guidance (random), 0.7 = balanced (default), >1.0 = very strong adherence to reference characteristics."
                )

            # Row 2: Reference Audio Processing Limits
            with gr.Row():
                with gr.Column():
                    max_speaker_audio_length = gr.Slider(
                        label="Max Speaker Reference Length (seconds)",
                        value=30,
                        minimum=3,
                        maximum=90,
                        step=1,
                        info="How much of the speaker reference audio to use. Model works best with 5-15 seconds. Maximum set to 90 seconds for safety. Default: 30s"
                    )
                with gr.Column():
                    max_emotion_audio_length = gr.Slider(
                        label="Max Emotion Reference Length (seconds)",
                        value=30,
                        minimum=3,
                        maximum=90,
                        step=1,
                        info="How much of the emotion reference audio to use. Model works best with 5-15 seconds. Maximum set to 90 seconds for safety. Default: 30s"
                    )

            # Row 3: Enable Sampling and Temperature
            with gr.Row():
                do_sample = gr.Checkbox(
                    label="Enable Sampling",
                    value=True,
                    info="When ON: Uses random sampling for natural, varied speech. When OFF: Always picks most likely tokens for consistent but potentially robotic output. Keep ON for natural speech."
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    info="Controls speech expressiveness. Higher (0.9-1.2) = more varied intonation and expression. Lower (0.3-0.7) = flatter but more stable speech. Default 0.8 is balanced."
                )

            # Row 4: Beam Search Beams and Max Tokens per Segment
            with gr.Row():
                num_beams = gr.Slider(
                    label="Beam Search Beams",
                    value=3,
                    minimum=1,
                    maximum=10,
                    step=1,
                    info="Explores multiple generation paths simultaneously. Higher (5-10) = better quality but slower. Lower (1-3) = faster but potentially worse quality. Default 3 balances speed and quality. Bigger also uses more VRAM."
                )
                initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                max_text_tokens_per_segment = gr.Textbox(
                    label="Max Tokens per Segment",
                    value=str(initial_value),
                    key="max_text_tokens_per_segment",
                    info=f"Splits long text into chunks for processing. Valid range: 20-{tts.cfg.gpt.max_text_tokens}. Smaller (80-120) = more natural pauses and consistent quality but slower. Larger (150-200) = faster but may have quality variations. Default: {initial_value}. Bigger value uses more VRAM."
                )

            # Row 5: Save as MP3 and Low Memory Mode
            with gr.Row():
                save_as_mp3 = gr.Checkbox(
                    label="Save as MP3",
                    value=MP3_AVAILABLE,
                    visible=MP3_AVAILABLE,
                    info="Save audio as MP3 format instead of WAV" if MP3_AVAILABLE else "Requires pydub: pip install pydub"
                )
                low_memory_mode = gr.Checkbox(
                    label="Low Memory Mode",
                    value=False,
                    info="Enable low memory mode for systems with limited GPU memory (inference will be slower)"
                )
                prevent_vram_accumulation = gr.Checkbox(
                    label="Prevent VRAM Accumulation",
                    value=False,
                    info="Reset beam search cache after each segment. Helps avoid VRAM growth at higher beams (e.g., 8). Slight performance impact."
                )

        with gr.Accordion("Preview Sentence Segmentation Results", open=True) as segments_settings:
            chapter_segmentation = gr.Checkbox(
                label="Chapter Segmentation",
                value=False,
                info="When enabled, segments are generated chapter-by-chapter using the detected chapter markers.",
            )
            segments_preview_mode = gr.Radio(
                label="Segmentation Preview Mode",
                choices=["Standard (instant)", "Experimental (progressive)"],
                value="Standard (instant)",
                info="Experimental mode streams preview rows with progress updates to avoid interface freezes on very long inputs.",
            )
            segments_preview = gr.Dataframe(
                headers=["Index", "Segment Content", "Token Count"],
                key="segments_preview",
                wrap=True,
                interactive=False,
            )
            processed_segments_state = gr.State(value=None)

        with gr.Accordion("Chapter Settings", open=False):
            enable_chapters = gr.Checkbox(
                label="Enable MP3 Chapters",
                value=False,
                info="When enabled, matches in the regex below will become chapter markers embedded into the exported MP3.",
            )
            chapter_regex_input = gr.Textbox(
                label="Chapter Start Regex",
                placeholder=r"e.g., ^Chapter \\d+",
                info="Provide a regular expression that matches the beginning of each chapter. Use multiline mode constructs such as ^ for line starts.",
            )
            chapter_error = gr.Markdown(value="", visible=False)
            chapter_preview = gr.Dataframe(
                headers=["Chapter #", "Title", "Start Char", "Preview Start"],
                interactive=False,
                visible=False,
                type="array",
            )
            chapter_status = gr.Markdown(value="", visible=False)

    with gr.Tab("Advanced Parameters"):
        gr.Markdown("### 🎯 Advanced Audio Generation Parameters")
        gr.Markdown("_Fine-tune generation parameters for expert control over audio synthesis._")

        with gr.Row():
            with gr.Column():
                mp3_bitrate = gr.Dropdown(
                    label="MP3 Bitrate",
                    choices=["128k", "192k", "256k", "320k"],
                    value="256k",
                    info="Audio quality when saving as MP3. 128k = smaller files but lower quality. 320k = best quality but larger files. 256k = good balance for most uses."
                )
            with gr.Column():
                latent_multiplier = gr.Slider(
                    label="Latent Length Multiplier",
                    value=1.72,
                    minimum=1.0,
                    maximum=3.0,
                    step=0.01,
                    info="Controls speech pacing speed. Higher (2.0-3.0) = slower, more stretched speech. Lower (1.0-1.5) = faster, more compressed speech. Default 1.72 is natural pacing."
                )

        with gr.Row():
            with gr.Column():
                top_p = gr.Slider(
                    label="Top-p (Nucleus Sampling)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.01,
                    info="Limits token selection to most probable options. Higher (0.9-1.0) = more varied and expressive speech. Lower (0.3-0.7) = more predictable, conservative speech. Default 0.8 balances variety and stability."
                )
            with gr.Column():
                top_k = gr.Slider(
                    label="Top-k",
                    minimum=0,
                    maximum=100,
                    value=30,
                    step=1,
                    info="Limits selection to k most probable tokens. Higher (50-100) = more speech variety. Lower (10-30) = more consistent speech. 0 = disabled. Default 30 avoids unlikely tokens while maintaining variety."
                )

        with gr.Row():
            with gr.Column():
                repetition_penalty = gr.Number(
                    label="Repetition Penalty",
                    precision=None,
                    value=10.0,
                    minimum=1.0,
                    maximum=20.0,
                    step=0.1,
                    info="Prevents speech from getting stuck in loops. Higher (10-15) = strongly avoids repetition. Lower (1-5) = allows natural repetition. Default 10.0 effectively prevents stuttering."
                )
            with gr.Column():
                length_penalty = gr.Number(
                    label="Length Penalty",
                    precision=None,
                    value=0.0,
                    minimum=-2.0,
                    maximum=2.0,
                    step=0.1,
                    info="Influences speech segment length. Positive (0.5-2.0) = longer segments. Negative (-2.0 to -0.5) = shorter segments. Zero = natural length based on content."
                )

        with gr.Row():
            with gr.Column():
                max_consecutive_silence = gr.Slider(
                    label="Max Consecutive Silent Tokens (0=disabled)",
                    value=0,
                    minimum=0,
                    maximum=100,
                    step=5,
                    info="Removes long pauses in speech. Higher (30-50) = allows longer natural pauses. Lower (5-20) = tighter, more continuous speech. 0 = no pause removal. Try 30 if output has awkward long silences."
                )
            with gr.Column():
                interval_silence = gr.Slider(
                    label="Silence Between Segments (ms)",
                    value=200,
                    minimum=0,
                    maximum=1000,
                    step=50,
                    info="Pause length between text segments. Higher (500-1000ms) = formal presentation style with clear breaks. Lower (50-200ms) = conversational flow. Default 200ms is natural for most content."
                )

        gr.Markdown("### 🎯 Emotion Control Parameters")
        with gr.Row():
            with gr.Column():
                apply_emo_bias = gr.Checkbox(
                    label="Apply Emotion Bias Correction",
                    value=True,
                    info="Prevents emotions from becoming too extreme or unnatural. Keeps emotional expression balanced and realistic. Recommended: Keep ON."
                )
            with gr.Column():
                max_emotion_sum = gr.Slider(
                    label="Max Total Emotion Strength",
                    value=0.8,
                    minimum=0.1,
                    maximum=2.0,
                    step=0.05,
                    info="Limits overall emotional intensity. Higher (1.0-2.0) = stronger emotions allowed. Lower (0.3-0.7) = more subtle emotions. Default 0.8 keeps emotions natural."
                )

        with gr.Row():
            with gr.Column():
                autoregressive_batch_size = gr.Slider(
                    label="Autoregressive Batch Size",
                    value=1,
                    minimum=1,
                    maximum=8,
                    step=1,
                    info="Generates multiple speech variations simultaneously. Higher (3-8) = more options, potentially better quality but much slower. 1 = single fast generation."
                )
            with gr.Column():
                max_mel_tokens = gr.Slider(
                    label="Max Mel Tokens",
                    value=1500,
                    minimum=50,
                    maximum=1815,
                    step=10,
                    info=f"Maximum speech length per segment. 1815 tokens ≈ 84 seconds. Lower values may cut off long segments. Default 1500 works for most content.",
                    key="max_mel_tokens"
                )

        gr.Markdown("### 🧠 Advanced Model Architecture Settings (Expert Only!)")
        gr.Markdown("⚠️ **WARNING**: These settings directly affect model internals. Only change if you understand the architecture!")

        with gr.Row():
            with gr.Column():
                semantic_layer = gr.Slider(
                    label="Semantic Feature Extraction Layer",
                    value=17,
                    minimum=1,
                    maximum=24,
                    step=1,
                    info="Which layer of the semantic model to use. Higher layers (15-20) = more expressive, emotion-aware speech. Lower layers (5-12) = clearer pronunciation. Default 17 balances both."
                )
            with gr.Column():
                cfm_cache_length = gr.Slider(
                    label="CFM Max Cache Sequence Length",
                    value=8192,
                    minimum=1024,
                    maximum=16384,
                    step=512,
                    info="Memory allocation for processing speech. Higher (12000-16000) = handles longer segments better but uses more VRAM. Lower (4000-8000) = less memory usage. Default 8192 works for most."
                )

        gr.Markdown("### 🎛️ Custom Emotion Bias Weights")
        gr.Markdown("Fine-tune individual emotion channel biases in normalize_emo_vec() when Apply Emotion Bias is enabled:")
        with gr.Row():
            emo_bias_joy = gr.Slider(label="Joy Bias", value=0.9375, minimum=0.5, maximum=1.5, step=0.0625,
                                     info="Adjusts how much joy/happiness is expressed. <1.0 = less joyful, >1.0 = more joyful")
            emo_bias_anger = gr.Slider(label="Anger Bias", value=0.875, minimum=0.5, maximum=1.5, step=0.0625,
                                       info="Adjusts anger intensity. <1.0 = less angry, >1.0 = more angry")
            emo_bias_sad = gr.Slider(label="Sadness Bias", value=1.0, minimum=0.5, maximum=1.5, step=0.0625,
                                     info="Adjusts sadness expression. <1.0 = less sad, >1.0 = more sad")
            emo_bias_fear = gr.Slider(label="Fear Bias", value=1.0, minimum=0.5, maximum=1.5, step=0.0625,
                                      info="Adjusts fear/anxiety expression. <1.0 = less fearful, >1.0 = more fearful")
        with gr.Row():
            emo_bias_disgust = gr.Slider(label="Disgust Bias", value=0.9375, minimum=0.5, maximum=1.5, step=0.0625,
                                         info="Adjusts disgust expression. <1.0 = less disgusted, >1.0 = more disgusted")
            emo_bias_depression = gr.Slider(label="Depression Bias", value=0.9375, minimum=0.5, maximum=1.5, step=0.0625,
                                           info="Adjusts melancholic/depressed tone. <1.0 = less depressed, >1.0 = more depressed")
            emo_bias_surprise = gr.Slider(label="Surprise Bias", value=0.6875, minimum=0.5, maximum=1.5, step=0.0625,
                                          info="Adjusts surprise/amazement expression. <1.0 = less surprised, >1.0 = more surprised")
            emo_bias_calm = gr.Slider(label="Calm Bias", value=0.5625, minimum=0.5, maximum=1.5, step=0.0625,
                                      info="Adjusts calm/neutral tone. <1.0 = less calm, >1.0 = more calm and peaceful")

        # Define parameter lists for function calls
        advanced_params = [
            do_sample, top_p, top_k, temperature,
            length_penalty, num_beams, repetition_penalty, max_mel_tokens,
            low_memory_mode, prevent_vram_accumulation,
        ]

        expert_params = [
            diffusion_steps, inference_cfg_rate, interval_silence,
            max_speaker_audio_length, max_emotion_audio_length,
            autoregressive_batch_size, apply_emo_bias, max_emotion_sum,
            latent_multiplier, max_consecutive_silence, mp3_bitrate
        ]

        model_params = [semantic_layer, cfm_cache_length,
                       emo_bias_joy, emo_bias_anger, emo_bias_sad, emo_bias_fear,
                       emo_bias_disgust, emo_bias_depression, emo_bias_surprise, emo_bias_calm]

        chapters_state = gr.State([])

    with gr.Tab("Extras"):
        gr.Markdown("### 🔧 Extras")
        with gr.Group():
            gr.Markdown("#### Isolate Vocals (Deep Extract)")
            gr.Markdown(
                "Upload a mixed track to extract vocals and instrumentals using the "
                "same Demucs two-stem workflow leveraged by the ComfyUI DeepExtract V2 "
                "pipeline."
            )
            if not DEMUCS_AVAILABLE:
                gr.Markdown(
                    "⚠️ `demucs` is not installed in this environment. Install it with ``pip install demucs`` "
                    "or `uv sync --extra vocal_isolation` to enable this tool.",
                    elem_classes=["warning-text"],
                )
            with gr.Row():
                vocal_isolation_file = gr.File(
                    label="Audio File",
                    file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                    type="filepath",
                )
                with gr.Column():
                    vocal_isolation_model = gr.Dropdown(
                        label="Demucs Model",
                        choices=[
                            "htdemucs",
                            "htdemucs_ft",
                            "htdemucs_6s",
                            "mdx",
                            "mdx_q",
                            "mdx_extra",
                            "mdx_extra_q",
                        ],
                        value="htdemucs",
                        info="Select the pre-trained Demucs model to run.",
                    )
                    vocal_isolation_device = gr.Radio(
                        label="Device",
                        choices=["Auto", "cuda", "cpu"],
                        value="Auto",
                        info="Choose the compute device for Demucs. Auto lets the library decide.",
                    )
            with gr.Row():
                vocal_isolation_segment = gr.Slider(
                    label="Segment Length (seconds)",
                    minimum=0,
                    maximum=60,
                    step=1,
                    value=0,
                    info="Optional chunk size passed to Demucs --segment. Use 0 to keep default.",
                )
                vocal_isolation_shifts = gr.Slider(
                    label="Shifts",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=1,
                    info="Number of prediction shifts to average. Higher = better quality, slower.",
                )
                vocal_isolation_overlap = gr.Slider(
                    label="Overlap",
                    minimum=0.0,
                    maximum=0.99,
                    step=0.01,
                    value=0.25,
                    info="Overlap ratio between segments. Matches Demucs --overlap.",
                )
            with gr.Row():
                vocal_isolation_mp3 = gr.Checkbox(
                    label="Export MP3",
                    value=False,
                    info="Convert extracted vocals to MP3 (requires pydub).",
                )
                vocal_isolation_mp3_bitrate = gr.Dropdown(
                    label="MP3 Bitrate",
                    choices=["128k", "160k", "192k", "256k", "320k"],
                    value="256k",
                )
                vocal_isolation_button = gr.Button(
                    "Isolate Vocals",
                    variant="primary",
                    interactive=DEMUCS_AVAILABLE,
                )
            vocal_isolation_status = gr.Markdown(value="", visible=False)
            with gr.Row():
                vocal_isolation_vocals = gr.Audio(
                    label="Extracted Vocals",
                    type="filepath",
                    interactive=False,
                    visible=False,
                )
                vocal_isolation_instrumental = gr.Audio(
                    label="Instrumental (No Vocals)",
                    type="filepath",
                    interactive=False,
                    visible=False,
                )
            vocal_isolation_logs = gr.Textbox(
                label="Demucs Log Output",
                value="",
                lines=6,
                interactive=False,
                visible=False,
            )
        with gr.Group():
            gr.Markdown("#### Merge MP3 Files")
            merge_mp3_inputs = gr.Files(
                label="MP3 Files",
                file_types=[".mp3"],
                file_count="multiple",
                type="filepath",
            )
            merge_keep_chapters = gr.Checkbox(
                label="Keep chapters",
                value=True,
                info=(
                    "When enabled, existing chapter markers are preserved and new markers are added at the start of each source MP3."
                    " Disable to replace all chapters with new ones at each MP3 boundary."
                ),
            )
            merge_output_name = gr.Textbox(
                label="Merged Output Filename (optional)",
                placeholder="Leave empty for auto-numbered merged_XXXX.mp3",
            )
            merge_button = gr.Button("Merge MP3s", variant="primary")
            merge_status = gr.Markdown(value="", visible=False)
            merged_audio_preview = gr.Audio(
                label="Merged MP3 Preview",
                type="filepath",
                interactive=False,
                visible=False,
            )

        with gr.Group():
            gr.Markdown("#### MP3 Chapter Editor")
            gr.Markdown("Load an MP3 to review its embedded chapters and rename them before saving.")
            with gr.Row():
                chapter_editor_file = gr.File(
                    label="MP3 File",
                    file_types=[".mp3"],
                    type="filepath",
                )
                chapter_load_button = gr.Button("Load Chapters")
            chapter_editor_table = gr.Dataframe(
                headers=["Title", "Start (mm:ss)"],
                datatype=["str", "str"],
                type="array",
                interactive=True,
                visible=False,
                row_count=(0, "dynamic"),
                col_count=(2, "fixed"),
            )
            chapter_editor_status = gr.Markdown(value="", visible=False)
            chapter_save_button = gr.Button("Save Chapter Titles", variant="primary", interactive=False)

        chapter_editor_state = gr.State({"path": "", "chapters": []})

    def process_media_upload(media_file, time_ranges):
        """Process uploaded media file and extract audio."""
        if media_file is None:
            return gr.update(visible=True), gr.update(visible=False), gr.update(value=None)

        try:
            # Extract audio from media file
            temp_audio = tempfile.mktemp(suffix=".wav")
            extracted_audio = extract_audio_from_media(media_file, temp_audio)

            if not extracted_audio:
                return gr.update(visible=True), gr.update(visible=False), gr.update(value=None)

            # If time ranges specified, extract and merge segments
            if time_ranges and time_ranges.strip():
                segments_audio = extract_time_ranges(extracted_audio, time_ranges)
                if segments_audio:
                    os.remove(extracted_audio)
                    extracted_audio = segments_audio

            # Hide upload panel, show audio panel with extracted audio
            return (
                gr.update(visible=False),  # Hide media upload
                gr.update(visible=True, value=extracted_audio),  # Show and update prompt_audio
                gr.update(value=extracted_audio)  # Return path
            )
        except Exception as e:
            print(f"Error processing media: {e}")
            return gr.update(visible=True), gr.update(visible=False), gr.update(value=None)

    def extract_audio_segments(media_file, time_ranges):
        """Extract specific time segments from uploaded media."""
        if media_file is None:
            return gr.update(), gr.update(), gr.update()

        try:
            # First extract full audio
            temp_audio = tempfile.mktemp(suffix=".wav")
            extracted_audio = extract_audio_from_media(media_file, temp_audio)

            if not extracted_audio:
                return gr.update(), gr.update(), gr.update()

            # Extract and merge segments if specified
            if time_ranges and time_ranges.strip():
                segments_audio = extract_time_ranges(extracted_audio, time_ranges)
                if segments_audio:
                    os.remove(extracted_audio)
                    extracted_audio = segments_audio

            # Update audio component
            return (
                gr.update(visible=False),  # Hide upload panel
                gr.update(visible=True, value=extracted_audio),  # Update audio
                gr.update(value=extracted_audio)  # Return path
            )
        except Exception as e:
            print(f"Error extracting segments: {e}")
            return gr.update(), gr.update(), gr.update()

    def clear_reference_audio():
        """Clear reference audio and show upload panel again."""
        return gr.update(visible=True), gr.update(visible=False, value=None)

    def load_audio_from_path_ui(audio_path):
        """Load audio from the specified file path."""
        if not audio_path:
            return gr.update(value=None), gr.update(value="Please enter a file path", visible=True)

        audio_path = audio_path.strip()
        if not os.path.exists(audio_path):
            return gr.update(value=None), gr.update(value=f"File not found: {audio_path}", visible=True)

        try:
            # Convert to acceptable format if needed
            temp_audio = tempfile.mktemp(suffix=".wav")
            extracted_audio = extract_audio_from_media(audio_path, temp_audio)

            if extracted_audio:
                return (
                    gr.update(visible=False),  # Hide upload panel
                    gr.update(visible=True, value=extracted_audio),  # Update prompt audio
                    gr.update(value="Audio loaded successfully!", visible=True)  # Status
                )
            else:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(value="Failed to load audio file", visible=True)
                )
        except Exception as e:
            return (
                gr.update(),
                gr.update(),
                gr.update(value=f"Error: {str(e)}", visible=True)
            )

    def load_text_file_contents(text_file, max_text_tokens_per_segment, preview_mode,
                                enable_chapters, chapter_regex, progress=gr.Progress(track_tqdm=True)):
        """Load text content from an uploaded .txt file."""
        if not text_file:
            yield (
                gr.update(),
                gr.update(),
                gr.update(value="", visible=False),
                gr.update(),
                gr.update(value="", visible=False),
                [],
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                None,
            )
            return

        file_path = text_file if isinstance(text_file, str) else getattr(text_file, "name", None)
        if not file_path or not os.path.exists(file_path):
            yield (
                gr.update(),
                gr.update(),
                gr.update(value="Text file not found.", visible=True),
                gr.update(),
                gr.update(value="", visible=False),
                [],
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                None,
            )
            return

        try:
            progress(0.0, desc="Reading text file")
            file_size = os.path.getsize(file_path)
            console_log(
                f"[Text loader] Reading {os.path.basename(file_path)} ({file_size / 1024:.1f} KiB)..."
            )
            if TEXT_FILE_SIZE_LIMIT and file_size > TEXT_FILE_SIZE_LIMIT:
                file_status = gr.update(
                    value=f"⚠️ Text file is larger than {TEXT_FILE_SIZE_LIMIT // (1024 * 1024)} MB. Please use a smaller file.",
                    visible=True,
                )
                yield (
                    gr.update(),
                    gr.update(),
                    file_status,
                    gr.update(),
                    gr.update(value="", visible=False),
                    [],
                    gr.update(value="", visible=False),
                    None,
                )
                return

            content = None
            encoding_used = None
            read_start = time.perf_counter()
            for encoding in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
                try:
                    with open(file_path, "r", encoding=encoding) as handle:
                        content = handle.read()
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
                encoding_used = "utf-8 (errors ignored)"

            read_elapsed = time.perf_counter() - read_start
            console_log(
                f"[Text loader] Loaded {len(content)} characters using {encoding_used} in {read_elapsed:.2f}s"
            )

            chapter_matches, chapter_error_msg, chapter_rows, table_visible = compute_chapter_preview_data(
                enable_chapters,
                content,
                chapter_regex,
            )

            chapter_table_update = gr.update(
                value=chapter_rows,
                visible=table_visible,
                type="array",
            ) if table_visible else gr.update(value=[], visible=False)

            chapter_error_update = gr.update(
                value=chapter_error_msg,
                visible=bool(chapter_error_msg) and table_visible,
            )

            max_tokens = parse_max_tokens(max_text_tokens_per_segment)
            token_count = len(tts.tokenizer.tokenize(content))
            estimated_segments = math.ceil(token_count / max_tokens) if max_tokens else None
            chapter_count = len(chapter_matches) if enable_chapters and chapter_matches else None
            token_message = format_token_analysis(token_count, estimated_segments, max_tokens, chapter_count)
            token_analysis_update = gr.update(value=token_message, visible=bool(token_message))

            file_status = gr.update(
                value=f"Loaded text file: {os.path.basename(file_path)} ({len(content)} characters)",
                visible=True,
            )

            progress(1.0, desc="File loaded")

            yield (
                gr.update(value=content),
                gr.update(value=[], visible=True, type="array"),
                file_status,
                chapter_table_update,
                chapter_error_update,
                chapter_matches,
                gr.update(value="", visible=False),
                token_analysis_update,
                None,
            )
            return
        except Exception as exc:
            yield (
                gr.update(),
                gr.update(),
                gr.update(value=f"Failed to load text file: {exc}", visible=True),
                gr.update(),
                gr.update(value="", visible=False),
                [],
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                None,
            )
            return

    def update_chapter_preview(enable_chapters, text, chapter_regex):
        """Update chapter preview table when text or regex changes."""
        matches, error_message, table_rows, table_visible = compute_chapter_preview_data(
            enable_chapters,
            text,
            chapter_regex,
        )

        table_update = gr.update(
            value=table_rows,
            visible=table_visible,
            type="array",
        ) if table_visible else gr.update(value=[], visible=False)

        error_update = gr.update(
            value=error_message,
            visible=bool(error_message) and table_visible,
        )

        status_update = gr.update(value="", visible=False)

        return table_update, error_update, matches, status_update



    def process_segments(text, max_text_tokens_per_segment, preview_mode,
                         use_chapter_segmentation, chapters_state,
                         progress=gr.Progress(track_tqdm=True)):
        sanitized_text = text or ""
        if not sanitized_text.strip():
            yield {
                segments_preview: gr.update(value=[], visible=True),
                processed_segments_state: None,
                token_analysis: gr.update(value="", visible=False),
            }
            return

        max_tokens = parse_max_tokens(max_text_tokens_per_segment)

        requested_chapter_mode = bool(use_chapter_segmentation)
        chapter_signature = build_chapter_signature(chapters_state) if requested_chapter_mode else ()
        chapter_chunks = build_segmentation_chapters(sanitized_text, chapters_state) if requested_chapter_mode else []
        active_chapter_mode = requested_chapter_mode and bool(chapter_chunks)

        if requested_chapter_mode and not active_chapter_mode:
            console_log(
                "[Segmentation] Chapter segmentation enabled but no chapters detected; processing full text instead."
            )

        active_chunks = chapter_chunks if active_chapter_mode else [{
            "title": "Full Text",
            "start": 0,
            "end": len(sanitized_text),
            "text": sanitized_text,
            "index": 0,
        }]

        try:
            overall_start = time.perf_counter()
            progress(0.0, desc="Preparing segmentation")
            log_prefix = "Segmentation preview (experimental)" if preview_mode == "Experimental (progressive)" else "Segmentation preview (standard)"
            console_log(
                f"[{log_prefix}] Preparing preview with max {max_tokens} tokens per segment"
            )
            if active_chapter_mode:
                console_log(
                    f"[{log_prefix}] Using chapter segmentation with {len(active_chunks)} chapter(s)"
                )

            chapter_results = []
            total_tokens = 0
            total_segments = 0

            for chunk_idx, chunk in enumerate(active_chunks):
                chapter_title = chunk.get("title") or f"Chapter {chunk_idx + 1}"
                chapter_text = chunk.get("text", "") or ""
                chapter_char_len = len(chapter_text)

                if not chapter_text.strip():
                    console_log(
                        f"[{log_prefix}] Chapter '{chapter_title}' is empty; skipping."
                    )
                    chapter_results.append({
                        "meta": chunk,
                        "segments": [],
                        "token_total": 0,
                    })
                    continue

                console_log(
                    f"[{log_prefix}] Tokenizing chapter {chunk_idx + 1}/{len(active_chunks)} '{chapter_title}' ({chapter_char_len} characters)"
                )
                token_start = time.perf_counter()
                chapter_tokens = tts.tokenizer.tokenize(chapter_text)
                token_elapsed = time.perf_counter() - token_start
                console_log(
                    f"[{log_prefix}] Chapter '{chapter_title}' tokenized into {len(chapter_tokens)} tokens in {token_elapsed:.2f}s"
                )

                split_start = time.perf_counter()
                split_segments = tts.tokenizer.split_segments(
                    chapter_tokens,
                    max_text_tokens_per_segment=max_tokens,
                )
                split_elapsed = time.perf_counter() - split_start
                console_log(
                    f"[{log_prefix}] Chapter '{chapter_title}' produced {len(split_segments)} segment(s) in {split_elapsed:.2f}s"
                )

                chapter_segment_rows = []
                chapter_token_total = 0
                for seg_idx, segment_tokens in enumerate(split_segments):
                    segment_str = ''.join(segment_tokens)
                    token_count = len(segment_tokens)
                    chapter_segment_rows.append({
                        "index": seg_idx,
                        "text": segment_str,
                        "token_count": token_count,
                    })
                    chapter_token_total += token_count

                total_tokens += chapter_token_total
                total_segments += len(chapter_segment_rows)
                chapter_results.append({
                    "meta": chunk,
                    "segments": chapter_segment_rows,
                    "token_total": chapter_token_total,
                })

                if active_chunks:
                    progress(
                        0.1 + 0.1 * (chunk_idx + 1) / len(active_chunks),
                        desc=f"Processed chapter {chunk_idx + 1}/{len(active_chunks)}",
                    )

            if total_segments == 0:
                progress(1.0, desc="No segments found")
                yield {
                    segments_preview: gr.update(value=[], visible=True, type="array"),
                    processed_segments_state: None,
                    token_analysis: gr.update(value="", visible=False),
                }
                return

            flattened_preview = []
            global_index = 0
            for chapter_data in chapter_results:
                for segment in chapter_data["segments"]:
                    flattened_preview.append([global_index, segment["text"], segment["token_count"]])
                    global_index += 1

            token_message = format_token_analysis(
                total_tokens,
                total_segments,
                max_tokens,
                chapter_count=len(active_chunks) if active_chapter_mode else None,
            )

            state_value = {
                "text": sanitized_text,
                "max_tokens": max_tokens,
                "segments": list(flattened_preview),
                "mode": preview_mode,
                "processed_at": time.perf_counter(),
                "chapter_mode": active_chapter_mode,
                "chapter_signature": chapter_signature if active_chapter_mode else (),
                "chapters": [
                    {
                        "title": chapter_data["meta"].get("title"),
                        "start": chapter_data["meta"].get("start"),
                        "end": chapter_data["meta"].get("end"),
                        "text": chapter_data["meta"].get("text"),
                    }
                    for chapter_data in chapter_results
                    if active_chapter_mode
                ],
                "total_tokens": total_tokens,
                "total_segments": total_segments,
            }

            if preview_mode == "Experimental (progressive)":
                stream_start = time.perf_counter()
                last_log_time = None
                preview_rows = []
                for idx, row in enumerate(flattened_preview):
                    preview_rows.append(row)
                    progress(
                        0.2 + 0.7 * ((idx + 1) / total_segments),
                        desc=f"Building preview ({idx + 1}/{total_segments})",
                    )
                    last_log_time = log_segmentation_progress(
                        log_prefix,
                        idx + 1,
                        total_segments,
                        stream_start,
                        last_log_time,
                    )
                    yield {
                        segments_preview: gr.update(value=list(preview_rows), visible=True, type="array"),
                    }

                progress(1.0, desc="Preview ready")
                log_segmentation_summary(
                    log_prefix,
                    total_segments,
                    total_tokens,
                    time.perf_counter() - overall_start,
                )

                yield {
                    processed_segments_state: state_value,
                    token_analysis: gr.update(value=token_message, visible=bool(token_message)),
                }
            else:
                progress(1.0, desc="Preview ready")
                log_segmentation_summary(
                    log_prefix,
                    total_segments,
                    total_tokens,
                    time.perf_counter() - overall_start,
                )
                yield {
                    segments_preview: gr.update(value=flattened_preview, visible=True, type="array"),
                    processed_segments_state: state_value,
                    token_analysis: gr.update(value=token_message, visible=bool(token_message)),
                }
        except Exception as exc:
            progress(1.0, desc="Failed to build preview")
            console_log(f"Error during segmentation preview: {exc}")
            yield {
                segments_preview: gr.update(value=[], visible=True),
                processed_segments_state: None,
                token_analysis: gr.update(value="", visible=False),
            }


    def invalidate_processed_segments(*_):
        """Clear stored segments when text or settings change."""
        return (
            gr.update(value=[], visible=True, type="array"),
            None,
            gr.update(value="", visible=False),
        )

    def generate_from_processed(
        segments_state,
        emo_control_method,
        prompt,
        text,
        save_used_audio,
        output_filename,
        emo_ref_path,
        emo_weight,
        vec1,
        vec2,
        vec3,
        vec4,
        vec5,
        vec6,
        vec7,
        vec8,
        emo_text,
        emo_random,
        max_text_tokens_per_segment,
        use_chapter_segmentation,
        enable_chapters,
        chapters_state,
        save_as_mp3,
        diffusion_steps,
        inference_cfg_rate,
        interval_silence,
        max_speaker_audio_length,
        max_emotion_audio_length,
        autoregressive_batch_size,
        apply_emo_bias,
        max_emotion_sum,
        latent_multiplier,
        max_consecutive_silence,
        mp3_bitrate,
        do_sample,
        top_p,
        top_k,
        temperature,
        length_penalty,
        num_beams,
        repetition_penalty,
        max_mel_tokens,
        low_memory_mode,
        prevent_vram_accumulation,
        semantic_layer,
        cfm_cache_length,
        emo_bias_joy,
        emo_bias_anger,
        emo_bias_sad,
        emo_bias_fear,
        emo_bias_disgust,
        emo_bias_depression,
        emo_bias_surprise,
        emo_bias_calm,
        progress=gr.Progress(),
    ):
        sanitized_text = text or ""
        if not sanitized_text.strip():
            raise gr.Error("Please enter text and process segments before generating audio.")

        max_tokens = parse_max_tokens(max_text_tokens_per_segment)

        if not segments_state or not isinstance(segments_state, dict):
            raise gr.Error("Please process segments before generating audio.")

        if (
            segments_state.get("text") != sanitized_text
            or segments_state.get("max_tokens") != max_tokens
        ):
            raise gr.Error("Text or segmentation settings have changed. Process segments again before generating.")

        state_chapter_mode = bool(segments_state.get("chapter_mode"))
        requested_chapter_mode = bool(use_chapter_segmentation)
        if state_chapter_mode != requested_chapter_mode:
            raise gr.Error("Chapter segmentation setting has changed. Process segments again before generating.")

        chapter_segments = None
        if state_chapter_mode:
            stored_signature = tuple(segments_state.get("chapter_signature") or ())
            current_signature = build_chapter_signature(chapters_state)
            if stored_signature != current_signature:
                raise gr.Error("Chapter boundaries have changed. Process segments again before generating.")
            chapter_segments = segments_state.get("chapters") or []

        return gen_single(
            emo_control_method,
            prompt,
            text,
            save_used_audio,
            output_filename,
            emo_ref_path,
            emo_weight,
            vec1,
            vec2,
            vec3,
            vec4,
            vec5,
            vec6,
            vec7,
            vec8,
            emo_text,
            emo_random,
            max_text_tokens_per_segment,
            chapter_segments,
            enable_chapters,
            chapters_state,
            save_as_mp3,
            diffusion_steps,
            inference_cfg_rate,
            interval_silence,
            max_speaker_audio_length,
            max_emotion_audio_length,
            autoregressive_batch_size,
            apply_emo_bias,
            max_emotion_sum,
            latent_multiplier,
            max_consecutive_silence,
            mp3_bitrate,
            do_sample,
            top_p,
            top_k,
            temperature,
            length_penalty,
            num_beams,
            repetition_penalty,
            max_mel_tokens,
            low_memory_mode,
            prevent_vram_accumulation,
            semantic_layer,
            cfm_cache_length,
            emo_bias_joy,
            emo_bias_anger,
            emo_bias_sad,
            emo_bias_fear,
            emo_bias_disgust,
            emo_bias_depression,
            emo_bias_surprise,
            emo_bias_calm,
            progress=progress,
        )

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )


    input_text_single.change(
        invalidate_processed_segments,
        inputs=[],
        outputs=[segments_preview, processed_segments_state, token_analysis]
    )

    max_text_tokens_per_segment.change(
        invalidate_processed_segments,
        inputs=[],
        outputs=[segments_preview, processed_segments_state, token_analysis]
    )

    segments_preview_mode.change(
        invalidate_processed_segments,
        inputs=[],
        outputs=[segments_preview, processed_segments_state, token_analysis]
    )

    chapter_segmentation.change(
        invalidate_processed_segments,
        inputs=[],
        outputs=[segments_preview, processed_segments_state, token_analysis]
    )

    text_file_upload.change(
        load_text_file_contents,
        inputs=[text_file_upload, max_text_tokens_per_segment, segments_preview_mode, enable_chapters, chapter_regex_input],
        outputs=[input_text_single, segments_preview, text_file_status, chapter_preview, chapter_error, chapters_state, chapter_status, token_analysis, processed_segments_state]
    )

    enable_chapters.change(
        update_chapter_preview,
        inputs=[enable_chapters, input_text_single, chapter_regex_input],
        outputs=[chapter_preview, chapter_error, chapters_state, chapter_status]
    )

    chapter_regex_input.change(
        update_chapter_preview,
        inputs=[enable_chapters, input_text_single, chapter_regex_input],
        outputs=[chapter_preview, chapter_error, chapters_state, chapter_status]
    )

    input_text_single.change(
        update_chapter_preview,
        inputs=[enable_chapters, input_text_single, chapter_regex_input],
        outputs=[chapter_preview, chapter_error, chapters_state, chapter_status]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[process_and_generate_button, generate_from_processed_button])

    # New UI callbacks
    media_upload.change(
        process_media_upload,
        inputs=[media_upload, time_ranges_input],
        outputs=[media_upload_group, prompt_audio, prompt_audio]
    )

    extract_button.click(
        extract_audio_segments,
        inputs=[media_upload, time_ranges_input],
        outputs=[media_upload_group, prompt_audio, prompt_audio]
    )

    prompt_audio.clear(
        clear_reference_audio,
        outputs=[media_upload_group, prompt_audio]
    )

    load_audio_button.click(
        load_audio_from_path_ui,
        inputs=[audio_path_input],
        outputs=[media_upload_group, prompt_audio, load_status]
    )

    process_segments_button.click(
        process_segments,
        inputs=[input_text_single, max_text_tokens_per_segment, segments_preview_mode, chapter_segmentation, chapters_state],
        outputs=[segments_preview, processed_segments_state, token_analysis],
    )

    process_and_generate_button.click(
        process_segments,
        inputs=[input_text_single, max_text_tokens_per_segment, segments_preview_mode, chapter_segmentation, chapters_state],
        outputs=[segments_preview, processed_segments_state, token_analysis],
    ).then(
        generate_from_processed,
        inputs=[processed_segments_state,
                emo_control_method,
                prompt_audio,
                input_text_single,
                save_used_audio,
                output_filename,
                emo_upload,
                emo_weight,
                vec1,
                vec2,
                vec3,
                vec4,
                vec5,
                vec6,
                vec7,
                vec8,
                emo_text,
                emo_random,
                max_text_tokens_per_segment,
                chapter_segmentation,
                enable_chapters,
                chapters_state,
                save_as_mp3,
                *expert_params,
                *advanced_params,
                *model_params,
        ],
        outputs=[output_audio, chapter_status, chapter_preview, chapters_state],
    )

    generate_from_processed_button.click(
        generate_from_processed,
        inputs=[processed_segments_state,
                emo_control_method,
                prompt_audio,
                input_text_single,
                save_used_audio,
                output_filename,
                emo_upload,
                emo_weight,
                vec1,
                vec2,
                vec3,
                vec4,
                vec5,
                vec6,
                vec7,
                vec8,
                emo_text,
                emo_random,
                max_text_tokens_per_segment,
                chapter_segmentation,
                enable_chapters,
                chapters_state,
                save_as_mp3,
                *expert_params,
                *advanced_params,
                *model_params,
        ],
        outputs=[output_audio, chapter_status, chapter_preview, chapters_state],
    )

    chapter_load_button.click(
        load_mp3_chapter_data,
        inputs=[chapter_editor_file],
        outputs=[chapter_editor_table, chapter_editor_status, chapter_editor_state, chapter_save_button],
    )

    chapter_save_button.click(
        save_mp3_chapter_titles,
        inputs=[chapter_editor_table, chapter_editor_state],
        outputs=[chapter_editor_status, chapter_editor_state, chapter_editor_table],
    )

    vocal_isolation_button.click(
        isolate_vocals_ui,
        inputs=[
            vocal_isolation_file,
            vocal_isolation_model,
            vocal_isolation_device,
            vocal_isolation_segment,
            vocal_isolation_shifts,
            vocal_isolation_overlap,
            vocal_isolation_mp3,
            vocal_isolation_mp3_bitrate,
        ],
        outputs=[
            vocal_isolation_status,
            vocal_isolation_vocals,
            vocal_isolation_instrumental,
            vocal_isolation_logs,
        ],
    )

    merge_button.click(
        merge_mp3_files,
        inputs=[merge_mp3_inputs, merge_keep_chapters, merge_output_name, mp3_bitrate],
        outputs=[merge_status, merged_audio_preview],
    )

    open_outputs_button.click(open_outputs_folder)



if __name__ == "__main__":
    demo.queue(20)
    demo.launch(
        share=cmd_args.share,
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True
    )
