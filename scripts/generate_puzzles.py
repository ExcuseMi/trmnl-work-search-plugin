#!/usr/bin/env python3
"""
Word Search Puzzle Generator (Simplified & Reliable)
- Supports 3500 puzzles per {grid_size}/{difficulty}
- Stores theme words locally to avoid repeated API calls
"""

import requests
import random
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
THEMES_FILE = Path("themes.json")
THEME_WORDS_CACHE = Path("theme_words.json")
GRID_SIZES = [8, 10, 12, 15]
DIFFICULTIES = ["easy", "medium", "hard"]
PUZZLES_PER_COMBO = 3500
OUTPUT_DIR = Path("data")

ALL_DIRECTIONS = [
    (0, 1),  # Right
    (1, 0),  # Down
    (1, 1),  # Down-Right
    (-1, 1),  # Up-Right
    (0, -1),  # Left
    (-1, 0),  # Up
    (-1, -1),  # Up-Left
    (1, -1)  # Down-Left
]

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# ------------------------------------------------------------
# Load themes
# ------------------------------------------------------------
def load_themes() -> List[str]:
    """Load themes from JSON file."""
    try:
        with open(THEMES_FILE, 'r') as f:
            themes = json.load(f)
        logging.info(f"Loaded {len(themes)} themes from {THEMES_FILE}")
        return themes
    except FileNotFoundError:
        logging.error(f"Themes file not found: {THEMES_FILE}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in {THEMES_FILE}")
        raise


# ------------------------------------------------------------
# Theme words cache management
# ------------------------------------------------------------
def load_theme_cache() -> Dict[str, List[str]]:
    """Load cached theme words from file."""
    if THEME_WORDS_CACHE.exists():
        try:
            with open(THEME_WORDS_CACHE, 'r') as f:
                cache = json.load(f)
            logging.info(f"Loaded theme cache with {len(cache)} themes")
            return cache
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Could not load theme cache: {e}")
            return {}
    return {}


def save_theme_cache(cache: Dict[str, List[str]]) -> None:
    """Save theme words cache to file."""
    try:
        with open(THEME_WORDS_CACHE, 'w') as f:
            json.dump(cache, f, indent=2)
        logging.info(f"Saved theme cache with {len(cache)} themes")
    except IOError as e:
        logging.error(f"Could not save theme cache: {e}")


def fetch_theme_words(theme: str) -> List[str]:
    """Fetch words related to a theme from Datamuse API."""
    try:
        url = f"https://api.datamuse.com/words?rel_trg={theme}&max=100"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        words = [item['word'].upper() for item in response.json()
                 if 'word' in item and item['word'].isalpha()]
        logging.debug(f"Fetched {len(words)} words for theme: {theme}")
        return words
    except requests.RequestException as e:
        logging.warning(f"Failed to fetch words for {theme}: {e}")
        return []
    except (KeyError, ValueError) as e:
        logging.warning(f"Invalid response for {theme}: {e}")
        return []


def update_theme_cache(themes: List[str]) -> Dict[str, List[str]]:
    """
    Update the theme cache by fetching missing themes.
    Returns the complete cache.
    """
    cache = load_theme_cache()
    new_themes_fetched = 0

    for theme in themes:
        # Check if theme already exists
        if theme in cache and cache[theme] and len(cache[theme]) >= 30:
            continue  # Skip, already cached with enough words

        words = fetch_theme_words(theme)
        if words:  # Cache whatever we get
            cache[theme] = words
            new_themes_fetched += 1
            logging.debug(f"Cached {len(words)} words for '{theme}'")

        # Respect API rate limits
        time.sleep(0.3)

        # Save periodically
        if new_themes_fetched % 20 == 0:
            save_theme_cache(cache)

    if new_themes_fetched > 0:
        save_theme_cache(cache)
        logging.info(f"Fetched words for {new_themes_fetched} new themes")

    return cache


# ------------------------------------------------------------
# Difficulty Profiles
# ------------------------------------------------------------
def get_difficulty_params(difficulty: str, grid_size: int) -> Dict:
    """Get parameters for a specific difficulty level."""
    if difficulty == "easy":
        # Easy: Simple, clean puzzles
        word_count = min(grid_size, 8)
        return {
            'difficulty_label': 'easy',
            'directions': [(0, 1), (1, 0)],  # Only Right and Down
            'backwards_ratio': 0.0,
            'word_count': word_count,
            'min_len': 4,
            'max_len': min(6, grid_size - 2),
            'placement_attempts': 100,
            'min_words_required': max(3, word_count * 0.7)
        }

    elif difficulty == "medium":
        # Medium: Moderate challenge
        word_count = int(grid_size * 0.65)
        if grid_size == 15:
            word_count = 9
        elif grid_size == 12:
            word_count = 8
        elif grid_size == 10:
            word_count = 6
        else:  # 8x8
            word_count = 5

        return {
            'difficulty_label': 'medium',
            'directions': [(0, 1), (1, 0), (1, 1), (-1, 1)],  # 4 directions
            'backwards_ratio': 0.08,
            'word_count': word_count,
            'min_len': 4,
            'max_len': min(8, grid_size - 2),
            'placement_attempts': 150,
            'min_words_required': max(4, word_count * 0.7)
        }

    else:  # hard
        # Hard: Maximum challenge but still playable
        if grid_size == 15:
            word_count = 7
            max_len = 10
        elif grid_size == 12:
            word_count = 6
            max_len = 8
        elif grid_size == 10:
            word_count = 5
            max_len = 7
        else:  # 8x8
            word_count = 4
            max_len = 6

        return {
            'difficulty_label': 'hard',
            'directions': ALL_DIRECTIONS,  # All 8 directions
            'backwards_ratio': 0.25,
            'word_count': word_count,
            'min_len': 4,
            'max_len': max_len,
            'placement_attempts': 200,
            'min_words_required': max(3, word_count * 0.7)
        }


# ------------------------------------------------------------
# Word Filtering
# ------------------------------------------------------------
def filter_words(words: List[str], min_len: int, max_len: int, target_count: int) -> List[str]:
    """Filter words by length and remove duplicates."""
    # Filter by length and alphabetic
    filtered = []
    seen = set()

    for w in words:
        w_upper = w.upper()
        if (min_len <= len(w_upper) <= max_len and
                w_upper.isalpha() and
                w_upper not in seen):
            seen.add(w_upper)
            filtered.append(w_upper)

    # Return enough words for selection
    return filtered[:target_count * 3]


# ------------------------------------------------------------
# Placement Helpers
# ------------------------------------------------------------
def can_place_word(grid: List[List[str]], word: str, r: int, c: int,
                   dy: int, dx: int, size: int) -> bool:
    """Check if a word can be placed at the given position."""
    # Check bounds
    end_r = r + (len(word) - 1) * dy
    end_c = c + (len(word) - 1) * dx
    if not (0 <= end_r < size and 0 <= end_c < size):
        return False

    # Check each cell
    for i, ch in enumerate(word):
        rr = r + i * dy
        cc = c + i * dx
        if grid[rr][cc] not in ('', ch):
            return False

    return True


def place_word(grid: List[List[str]], word: str, r: int, c: int, dy: int, dx: int) -> None:
    """Place a word on the grid."""
    for i, ch in enumerate(word):
        grid[r + i * dy][c + i * dx] = ch


# ------------------------------------------------------------
# Puzzle Generator - SIMPLIFIED AND RELIABLE
# ------------------------------------------------------------
def generate_puzzle(theme: str, size: int, params: Dict,
                    available_words: List[str], puzzle_id: str) -> Optional[Dict]:
    """Generate a single word search puzzle."""
    grid = [['' for _ in range(size)] for _ in range(size)]
    dirs = params['directions']
    attempts = params['placement_attempts']

    # Filter and select words
    filtered = [w for w in available_words
                if params['min_len'] <= len(w) <= params['max_len']]

    if len(filtered) < params['word_count']:
        return None

    # Select unique words
    base_words = []
    seen = set()
    for w in filtered:
        if w not in seen:
            seen.add(w)
            base_words.append(w)
        if len(base_words) >= params['word_count']:
            break

    if len(base_words) < params['word_count']:
        return None

    placed_words = []
    solution = []
    backwards_target = int(params['word_count'] * params['backwards_ratio'])
    backwards_used = 0

    for word in base_words:
        original_word = word
        placed_word = word

        # Possibly reverse the word
        if backwards_used < backwards_target and random.random() < params['backwards_ratio']:
            placed_word = word[::-1]
            backwards_used += 1

        placed = False

        # Try to place the word
        for _ in range(attempts):
            dy, dx = random.choice(dirs)
            r = random.randint(0, size - 1)
            c = random.randint(0, size - 1)

            if can_place_word(grid, placed_word, r, c, dy, dx, size):
                place_word(grid, placed_word, r, c, dy, dx)
                pos = r * size + c
                dir_idx = ALL_DIRECTIONS.index((dy, dx))

                solution.append(f"{pos};{dir_idx};{len(original_word)};{original_word}")
                placed_words.append(original_word)
                placed = True
                break

        # If not placed with random attempts, try systematic placement
        if not placed:
            # Try all possible positions (within reason)
            max_systematic = min(100, size * size * len(dirs))
            for _ in range(max_systematic):
                dy, dx = random.choice(dirs)
                r = random.randint(0, size - 1)
                c = random.randint(0, size - 1)

                if can_place_word(grid, placed_word, r, c, dy, dx, size):
                    place_word(grid, placed_word, r, c, dy, dx)
                    pos = r * size + c
                    dir_idx = ALL_DIRECTIONS.index((dy, dx))

                    solution.append(f"{pos};{dir_idx};{len(original_word)};{original_word}")
                    placed_words.append(original_word)
                    placed = True
                    break

    # Check if we placed enough words
    if len(placed_words) < params['min_words_required']:
        logging.debug(f"Only placed {len(placed_words)} words in {puzzle_id}")
        return None

    # Verify solution before filling grid
    if not verify_solution(grid, solution, size):
        logging.debug(f"Solution verification failed for {puzzle_id} before filling")
        return None

    # Fill remaining empty cells with random letters
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for r in range(size):
        for c in range(size):
            if not grid[r][c]:
                grid[r][c] = random.choice(letters)

    # Final verification
    if not verify_solution(grid, solution, size):
        logging.debug(f"Solution verification failed for {puzzle_id} after filling")
        return None

    # Remove any duplicate words (shouldn't happen but just in case)
    unique_words = []
    seen_words = set()
    for w in placed_words:
        if w not in seen_words:
            seen_words.add(w)
            unique_words.append(w)

    # Flatten grid to string
    flat_grid = ''.join(''.join(row) for row in grid)

    return {
        'id': puzzle_id,
        'theme': theme,
        'grid': flat_grid,
        'solution': ','.join(solution),
        'gridSize': size,
        'difficulty': params['difficulty_label'],
        'wordCount': len(unique_words),
        'wordlist': sorted(unique_words, key=lambda w: (len(w), w.lower()))
    }


def verify_solution(grid: List[List[str]], solution: List[str], size: int) -> bool:
    """Verify that all words in the solution can be found in the grid."""
    try:
        for sol in solution:
            parts = sol.split(';')
            if len(parts) != 4:
                return False

            pos_str, dir_idx_str, length_str, original_word = parts

            # Parse values
            try:
                pos = int(pos_str)
                dir_idx = int(dir_idx_str)
                length = int(length_str)
            except ValueError:
                return False

            # Check direction index
            if dir_idx < 0 or dir_idx >= len(ALL_DIRECTIONS):
                return False

            dy, dx = ALL_DIRECTIONS[dir_idx]

            # Calculate start position
            start_row = pos // size
            start_col = pos % size

            # Check bounds for entire word
            end_row = start_row + (length - 1) * dy
            end_col = start_col + (length - 1) * dx
            if not (0 <= end_row < size and 0 <= end_col < size):
                return False

            # Reconstruct word from grid
            reconstructed = ''
            for i in range(length):
                r = start_row + i * dy
                c = start_col + i * dx
                reconstructed += grid[r][c]

            # Check if it matches original word or its reverse
            if reconstructed != original_word and reconstructed != original_word[::-1]:
                return False

        return True
    except Exception as e:
        logging.debug(f"Error in verify_solution: {e}")
        return False


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    """Main function to generate puzzles."""
    # Load themes
    themes = load_themes()

    # Update theme cache
    theme_cache = update_theme_cache(themes)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate puzzles for each grid size and difficulty
    for size in GRID_SIZES:
        for diff in DIFFICULTIES:
            folder = OUTPUT_DIR / str(size) / diff
            folder.mkdir(parents=True, exist_ok=True)

            params = get_difficulty_params(diff, size)
            file_counter = 1
            puzzles_made = 0
            failed_attempts = 0
            max_failed_attempts = 200

            logging.info(f"Generating {PUZZLES_PER_COMBO} puzzles for size={size}, difficulty={diff}")

            # Pre-filter available themes
            available_themes = []
            for theme in themes:
                words = theme_cache.get(theme, [])
                if not words:
                    continue

                usable = filter_words(words, params['min_len'], params['max_len'], params['word_count'])
                if len(usable) >= params['word_count']:
                    available_themes.append((theme, usable))

            if not available_themes:
                logging.error(f"No themes with enough words for {size}-{diff}")
                continue

            while puzzles_made < PUZZLES_PER_COMBO and failed_attempts < max_failed_attempts:
                # Randomly select a theme
                theme, usable_words = random.choice(available_themes)

                # Generate puzzle
                puzzle_id = f"{size}-{diff}-{file_counter}"

                try:
                    puzzle = generate_puzzle(theme, size, params, usable_words, puzzle_id)

                    if not puzzle:
                        failed_attempts += 1
                        file_counter += 1  # Increment even on failure

                        if failed_attempts % 50 == 0:
                            logging.warning(f"{failed_attempts} failed attempts for {size}-{diff}")
                        continue

                    # Additional validation
                    if len(puzzle['grid']) != size * size:
                        failed_attempts += 1
                        file_counter += 1
                        continue

                    if len(puzzle['wordlist']) != len(set(puzzle['wordlist'])):
                        failed_attempts += 1
                        file_counter += 1
                        continue

                    if len(puzzle['wordlist']) < params['min_words_required']:
                        failed_attempts += 1
                        file_counter += 1
                        continue

                    # Save puzzle
                    output_file = folder / f"{file_counter}.json"
                    with open(output_file, 'w') as f:
                        json.dump(puzzle, f, separators=(',', ':'))

                    puzzles_made += 1
                    file_counter += 1
                    failed_attempts = 0  # Reset on success

                    if puzzles_made % 100 == 0:
                        logging.info(f"  Generated {puzzles_made}/{PUZZLES_PER_COMBO} puzzles")
                        logging.info(f"  Success rate: {puzzles_made / (puzzles_made + failed_attempts):.1%}")

                except Exception as e:
                    logging.error(f"Error generating puzzle {puzzle_id}: {e}")
                    failed_attempts += 1
                    file_counter += 1
                    continue

            if failed_attempts >= max_failed_attempts:
                logging.error(f"Too many failed attempts for {size}-{diff}, generated {puzzles_made} puzzles")
            else:
                logging.info(f"Completed: size={size}, difficulty={diff} - {puzzles_made} puzzles")

    logging.info("Generation complete!")


if __name__ == "__main__":
    main()