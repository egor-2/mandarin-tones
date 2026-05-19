#!/usr/bin/env python3
"""Chinese Tone Guessing Game — two-process version with prefetch."""

import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import tempfile
import termios
import time
import tty

import pypinyin
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEQUE_PATH = os.path.join(BASE_DIR, "deque.txt")
DEQUE_MAX = 50
DEFAULT_SPEED = 0.8
NUM_AUDIO_VARIANTS = 5
TOKENS_PER_SYLLABLE = 10
DEBUG = "--debug" in sys.argv


# ── Worker process (model + generation) ─────────────────────────────
def audio_worker(req_q: mp.Queue, resp_q: mp.Queue, debug: bool):
    """Runs in a separate process. Loads model, generates audio on request."""
    from mlx_audio.tts.utils import load_model

    print("[worker] Loading model...")
    model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
    print("[worker] Model loaded.")
    resp_q.put(("model_ready",))

    while True:
        msg = req_q.get()
        if msg is None:
            break

        tag = msg[0]

        if tag == "batch_same":
            _, req_id, phrase, count, max_tokens = msg
            texts = [phrase] * count
            voices = ["Chelsie"] * count
            paths = []
            for result in model.batch_generate(texts=texts, voices=voices, lang_code="chinese",
                                               max_tokens=max_tokens):
                f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(f.name, result.audio, 24000)
                f.close()
                paths.append(f.name)
                if debug:
                    print(f"  [dbg] variant {result.sequence_idx}: {result.token_count}/{max_tokens} tokens, "
                          f"{result.audio_duration}, {result.samples} samples, "
                          f"{result.processing_time_seconds:.2f}s, {result.peak_memory_usage:.1f}MB")
            if debug:
                print(f"  [dbg] generated {len(paths)} variants for '{phrase}'")
            resp_q.put(("audio_ready", req_id, paths))

        elif tag == "batch_different":
            _, req_id, phrases, max_tokens_list = msg
            if not phrases:
                resp_q.put(("alts_ready", req_id, []))
                continue
            voices = ["Chelsie"] * len(phrases)
            mt = max(max_tokens_list) if max_tokens_list else 40
            paths = []
            for result in model.batch_generate(texts=phrases, voices=voices, lang_code="chinese",
                                               max_tokens=mt):
                f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(f.name, result.audio, 24000)
                f.close()
                paths.append(f.name)
                if debug:
                    print(f"  [dbg] alt {result.sequence_idx}: '{phrases[result.sequence_idx]}' "
                          f"{result.token_count}/{mt} tokens, {result.audio_duration}")
            if debug:
                print(f"  [dbg] generated {len(paths)} alternative audios")
            resp_q.put(("alts_ready", req_id, paths))


# ── Raw terminal input ──────────────────────────────────────────────
def read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def read_input(show_prompt=True):
    """Space=again  Enter=new  Digits+Enter=guess  Ctrl-C/D=quit."""
    buf = ""
    if show_prompt:
        sys.stdout.write("\n> ")
        sys.stdout.flush()
    while True:
        ch = read_key()
        if ch in ("\x03", "\x04"):
            return ("quit", None)
        if ch == " " and not buf:
            return ("again", None)
        if ch in ("\r", "\n"):
            sys.stdout.write("\n")
            sys.stdout.flush()
            if not buf:
                return ("new", None)
            digits = buf.replace("-", "").replace(" ", "")
            if digits and all(c in "012345" for c in digits):
                return ("tones", digits)
            print(f'  "{buf}" not recognized. Use digits 0-5 for tones.')
            buf = ""
            sys.stdout.write("> ")
            sys.stdout.flush()
            continue
        if ch in ("\x7f", "\x08"):
            if buf:
                buf = buf[:-1]
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        if ch.isprintable():
            buf += ch
            sys.stdout.write(ch)
            sys.stdout.flush()


# ── Deque management ────────────────────────────────────────────────
def load_deque() -> list:
    if not os.path.exists(DEQUE_PATH):
        return []
    entries = []
    for line in open(DEQUE_PATH):
        line = line.strip()
        if not line:
            continue
        try:
            level_str, idx_str = line.split(",", 1)
            entries.append((int(level_str), int(idx_str)))
        except ValueError:
            pass
    return entries


def save_deque(deque: list):
    with open(DEQUE_PATH, "w") as f:
        for level, idx in deque:
            f.write(f"{level},{idx}\n")


# ── Phrase cards ────────────────────────────────────────────────────
def load_cards(level: int) -> list:
    path = os.path.join(BASE_DIR, f"level{level}.json")
    with open(path) as f:
        return json.load(f)


def format_card(card: dict) -> str:
    return "\n".join([
        f"  Hanzi:   {card['hanzi']}",
        f"  Pinyin:  {card['pinyin']}",
        f"  Tones:   {card['tones']}",
        f"  Meaning: {card['meaning']}",
    ])


def play_wav(path: str, speed: float):
    subprocess.run(["afplay", "-r", str(speed), path])


def cleanup_audio(paths: list[str]):
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def print_controls():
    print("  [Space]=replay  [Enter]=new  [0-5]+Enter=guess  [Ctrl+C]=quit")


def get_pinyin(phrase: str) -> str:
    parts = pypinyin.pinyin(phrase, style=pypinyin.Style.TONE)
    return "".join(p[0] for p in parts)


# ── Tone alternatives ───────────────────────────────────────────────
def load_alternatives() -> dict:
    path = os.path.join(BASE_DIR, "alternatives.json")
    with open(path) as f:
        return json.load(f)


def find_alternatives(phrase: str, original_tones: list[int], alts_dict: dict, n: int = 5) -> list[tuple]:
    """Find n closest tone alternatives by Hamming distance.

    Returns list of (distance, tone_tuple, alt_phrase).
    """

    def available_tones(ch):
        if ch not in alts_dict:
            return set()
        return {i + 1 for i in range(4) if alts_dict[ch][i] is not None}

    def char_at_tone(ch, tone):
        if ch not in alts_dict:
            return None
        if 1 <= tone <= 4:
            return alts_dict[ch][tone - 1]
        return None

    def build_alt(tone_seq):
        result = []
        for ch, t in zip(phrase, tone_seq):
            c = char_at_tone(ch, t)
            if c is None:
                return None
            result.append(c)
        return "".join(result)

    chars = list(phrase)
    orig = original_tones
    candidates = []
    seen = set()

    # Single flips (distance 1)
    for i, ch in enumerate(chars):
        for t in sorted(available_tones(ch)):
            if t == orig[i]:
                continue
            new_tones = orig[:]
            new_tones[i] = t
            key = tuple(new_tones)
            if key not in seen:
                alt = build_alt(new_tones)
                if alt is not None:
                    seen.add(key)
                    candidates.append((1, key, alt))

    # Double flips (distance 2) if needed
    if len(candidates) < n:
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                for ti in sorted(available_tones(chars[i])):
                    if ti == orig[i]:
                        continue
                    for tj in sorted(available_tones(chars[j])):
                        if tj == orig[j]:
                            continue
                        new_tones = orig[:]
                        new_tones[i] = ti
                        new_tones[j] = tj
                        key = tuple(new_tones)
                        if key not in seen:
                            alt = build_alt(new_tones)
                            if alt is not None:
                                seen.add(key)
                                candidates.append((2, key, alt))
                        if len(candidates) >= n * 3:
                            break
                    if len(candidates) >= n * 3:
                        break
                if len(candidates) >= n * 3:
                    break
            if len(candidates) >= n * 3:
                break

    candidates.sort(key=lambda x: x[0])
    return candidates[:n]


# ── Word bundle: card + audio + alternatives ────────────────────────
class WordBundle:
    """Holds all data for one word: card, main audio paths, alt data + audio paths."""

    def __init__(self, idx, card, alts_data):
        self.idx = idx
        self.card = card
        self.alts_data = alts_data        # list of (dist, tone_tuple, alt_phrase)
        self.main_paths = None             # filled when audio_ready received
        self.alt_paths = None              # filled when alts_ready received
        self.audio_index = 0

    @property
    def phrase(self):
        return self.card["hanzi"]

    @property
    def main_ready(self):
        return self.main_paths is not None

    @property
    def alts_ready(self):
        return self.alt_paths is not None

    def cleanup(self):
        if self.main_paths:
            cleanup_audio(self.main_paths)
            self.main_paths = None
        if self.alt_paths:
            cleanup_audio(self.alt_paths)
            self.alt_paths = None


# ── Main process (game logic + UI) ──────────────────────────────────
def main():
    speed = DEFAULT_SPEED

    # Ask for level
    while True:
        try:
            raw = input("Level? (0, 1, or 2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if raw in ("0", "1", "2"):
            level = int(raw)
            break
        print("Please enter 0, 1, or 2.")

    cards = load_cards(level)
    deque = load_deque()
    alts_dict = load_alternatives()

    # Start worker
    req_q = mp.Queue()
    resp_q = mp.Queue()
    worker = mp.Process(target=audio_worker, args=(req_q, resp_q, DEBUG), daemon=True)
    worker.start()

    # Wait for model to load
    resp_q.get()  # ("model_ready",)
    print("Ready!\n")

    # ── helpers ──
    req_counter = 0

    def next_req_id():
        nonlocal req_counter
        req_counter += 1
        return req_counter

    # Maps req_id -> WordBundle
    pending = {}

    def pick_card():
        used = {(l, i) for l, i in deque if l == level}
        available = [i for i in range(len(cards)) if (level, i) not in used]
        if not available:
            available = list(range(len(cards)))
        idx = random.choice(available)
        return idx, cards[idx]

    def request_word(idx, card) -> WordBundle:
        """Compute alternatives, send both requests, return a WordBundle."""
        phrase = card["hanzi"]
        orig_tones = [int(t) for t in card["tones"].split("-")]
        alts_data = find_alternatives(phrase, orig_tones, alts_dict, n=5)

        # Request main audio
        main_id = next_req_id()
        max_tokens = len(phrase) * TOKENS_PER_SYLLABLE
        req_q.put(("batch_same", main_id, phrase, NUM_AUDIO_VARIANTS, max_tokens))

        # Request alternatives audio
        alt_id = next_req_id()
        alt_phrases = [a[2] for a in alts_data]
        alt_max_tokens = [len(p) * TOKENS_PER_SYLLABLE for p in alt_phrases]
        req_q.put(("batch_different", alt_id, alt_phrases, alt_max_tokens))

        bundle = WordBundle(idx, card, alts_data)
        pending[main_id] = ("main", bundle)
        pending[alt_id] = ("alts", bundle)
        return bundle

    def drain_responses():
        """Drain all available responses from the queue (non-blocking)."""
        while True:
            try:
                resp = resp_q.get_nowait()
            except Exception:
                break
            _handle_response(resp)

    def _handle_response(resp):
        tag, req_id, paths = resp
        if req_id not in pending:
            cleanup_audio(paths)
            return
        kind, bundle = pending.pop(req_id)
        if kind == "main":
            bundle.main_paths = paths
        elif kind == "alts":
            bundle.alt_paths = paths

    def wait_for(bundle, attr):
        """Block until bundle.attr is not None."""
        while getattr(bundle, attr) is None:
            resp = resp_q.get()
            _handle_response(resp)

    current = None
    prefetch = None
    guessed = False

    def cleanup(*_):
        save_deque(deque)
        if current:
            current.cleanup()
        if prefetch:
            prefetch.cleanup()
        req_q.put(None)
        print("\nDeque saved. Goodbye!")
        sys.exit(0)

    def activate_word(bundle):
        nonlocal current, guessed
        if current:
            current.cleanup()
        current = bundle
        deque.append((level, bundle.idx))
        if len(deque) > DEQUE_MAX:
            deque.pop(0)
        guessed = False
        bundle.audio_index = 0

        # Wait for main audio if not ready
        wait_for(bundle, "main_paths")
        play_wav(bundle.main_paths[0], speed)
        bundle.audio_index = 1
        print_controls()

    def start_prefetch():
        nonlocal prefetch
        if prefetch:
            prefetch.cleanup()
        idx, card = pick_card()
        prefetch = request_word(idx, card)

    # ── Generate first word ──
    first_idx, first_card = pick_card()
    first_bundle = request_word(first_idx, first_card)
    print(f"🔊 Generating audio...")
    activate_word(first_bundle)
    start_prefetch()

    # ── game loop ──
    show_prompt = True
    while True:
        # Drain any responses that arrived while user was thinking
        drain_responses()

        kind, val = read_input(show_prompt)
        show_prompt = True

        if kind == "quit":
            cleanup()

        elif kind == "new":
            print(f"🔊 Loading next phrase...")
            activate_word(prefetch)
            prefetch = None
            start_prefetch()

        elif kind == "again":
            if current is None:
                print("No phrase yet. Press Enter.")
            else:
                idx = current.audio_index % len(current.main_paths)
                play_wav(current.main_paths[idx], speed)
                current.audio_index = (current.audio_index + 1) % len(current.main_paths)
                show_prompt = False

        elif kind == "tones":
            if current is None:
                print("No phrase yet. Press Enter.")
                continue
            if guessed:
                print("Too late! Already guessed. Enter=new  Space=replay")
                continue

            guessed = True
            correct_tones = current.card["tones"].replace("-", "")
            guess = val.replace("5", "0")
            correct = correct_tones.replace("5", "0")

            if guess == correct:
                print("\n✅ Correct!")
            else:
                print("\n❌ Not quite!")

            print(format_card(current.card))

            # Wait for alt audio if not ready yet
            wait_for(current, "alt_paths")

            # Build menu items
            phrase = current.phrase
            orig_pinyin = get_pinyin(phrase)

            menu_items = []  # (label, pinyin, play_fn)
            menu_items.append(("Space", orig_pinyin,
                               lambda: play_wav(current.main_paths[0], speed)))
            for i, (dist, tone_tuple, alt_phrase) in enumerate(current.alts_data):
                alt_pinyin = get_pinyin(alt_phrase)
                idx_i = i  # capture
                menu_items.append((str(i + 1), alt_pinyin,
                                   (lambda ii: lambda: play_wav(current.alt_paths[ii], speed)
                                    if ii < len(current.alt_paths) else lambda: None)(idx_i)))

            # ANSI codes
            WHITE_BG = "\033[47m\033[30m"  # white bg, black text
            GRAY_BG = "\033[100m\033[97m"  # gray bg, white text
            RESET = "\033[0m"

            # Print menu
            print("\nPlay again, or play alternatives:")
            for i, (key, pinyin, _) in enumerate(menu_items):
                bg = WHITE_BG if i == 0 else GRAY_BG
                print(f"  {bg} {key:>5}  {pinyin} {RESET}")
            n_alts = len(current.alts_data)
            print(f"\n  [Space/1-{n_alts}] to play, [r]=random, [Enter] for next word, [Ctrl+C] to quit")

            # Interactive loop: Space/1-5 plays, Enter exits to next word
            while True:
                ch = read_key()
                if ch in ("\x03", "\x04"):
                    cleanup()
                if ch in ("\r", "\n"):
                    break
                if ch == " ":
                    idx = current.audio_index % len(current.main_paths)
                    play_wav(current.main_paths[idx], speed)
                    current.audio_index = (current.audio_index + 1) % len(current.main_paths)
                elif ch == "r" and n_alts > 0:
                    if random.random() < 0.5:
                        pick = menu_items[0]
                    else:
                        pick = random.choice(menu_items[1:])
                    pick[2]()
                elif ch.isdigit() and 1 <= int(ch) <= n_alts:
                    menu_items[int(ch)][2]()

            # Enter pressed -> next word
            print(f"\n🔊 Loading next phrase...")
            activate_word(prefetch)
            prefetch = None
            start_prefetch()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        if os.path.exists(DEQUE_PATH):
            print("(deque preserved)")
        raise
