#!/usr/bin/env python3
"""
Word Search Puzzle Generator (Enhanced Difficulty Tiers)
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
        url = f"https://api.datamuse.com/words?rel_trg={theme}&max=50"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        words = [item['word'].upper() for item in response.json()
                 if 'word' in item and item['word'].isalpha()]
        logging.info(f"Fetched {len(words)} words for theme: {theme}")
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
        # Check if theme already exists and has enough words
        if theme in cache:
            continue  # Skip, already cached with enough words

        words = fetch_theme_words(theme)
        if words and len(words) >= 20:  # Only cache if we got enough words
            cache[theme] = words
            new_themes_fetched += 1
        else:
            cache[theme] = []
            logging.warning(f"Theme '{theme}' has only {len(words) if words else 0} words")

        # Respect API rate limits
        time.sleep(0.3)

        # Save periodically
        if new_themes_fetched % 10 == 0:
            save_theme_cache(cache)

    if new_themes_fetched > 0:
        save_theme_cache(cache)
        logging.info(f"Fetched words for {new_themes_fetched} new themes")

    return cache


# ------------------------------------------------------------
# Difficulty Profiles - ADJUSTED FOR 15-HARD
# ------------------------------------------------------------
def get_difficulty_params(difficulty: str, grid_size: int) -> Dict:
    """Get parameters for a specific difficulty level."""
    if difficulty == "easy":
        # Easy: Simple, clean puzzles
        word_count = min(grid_size, 8)  # Max 8 words even for 15x15
        return {
            'difficulty_label': 'easy',
            'directions': [(0, 1), (1, 0)],  # Only Right and Down
            'backwards_ratio': 0.0,
            'word_count': word_count,
            'min_len': 4,
            'max_len': min(6, grid_size - 2),  # Shorter max length
            'placement_attempts': 150,
            'overlap_min': 0,
            'overlap_max': 1,  # Very little overlap
            'density_target': 0.25,  # Low density
            'min_words_required': max(3, word_count * 0.8)  # Must place most words
        }

    elif difficulty == "medium":
        # Medium: Moderate challenge, good mix
        # Scale word count with grid size: ~65% of grid size
        word_count = int(grid_size * 0.65)
        if grid_size == 15:
            word_count = 9  # 15 * 0.65 ≈ 9.75, round to 9
        elif grid_size == 12:
            word_count = 8  # 12 * 0.65 ≈ 7.8, round to 8
        elif grid_size == 10:
            word_count = 6  # 10 * 0.65 = 6.5, round to 6
        else:  # 8x8
            word_count = 5  # 8 * 0.65 ≈ 5.2, round to 5

        return {
            'difficulty_label': 'medium',
            'directions': [(0, 1), (1, 0), (1, 1), (-1, 1)],  # 4 directions
            'backwards_ratio': 0.08,  # Reduced from 0.10
            'word_count': word_count,
            'min_len': 4,
            'max_len': min(8, grid_size - 2),  # Reasonable max length
            'placement_attempts': 250,
            'overlap_min': 0,
            'overlap_max': 2,  # Moderate overlap
            'density_target': 0.35,  # Moderate density
            'min_words_required': max(4, word_count * 0.75)
        }

    else:  # hard
        # Hard: Maximum challenge but still playable
        # Scale word count: ~50% of grid size for hard (fewer but longer words)
        if grid_size == 15:
            word_count = 7  # Fewer words, but 8 directions + backwards
            max_len = 11
        elif grid_size == 12:
            word_count = 6
            max_len = 9
        elif grid_size == 10:
            word_count = 5
            max_len = 8
        else:  # 8x8
            word_count = 4
            max_len = 7

        return {
            'difficulty_label': 'hard',
            'directions': ALL_DIRECTIONS,  # All 8 directions
            'backwards_ratio': 0.25,  # Reduced from 0.30
            'word_count': word_count,
            'min_len': 4,  # Increased from 3 for better words
            'max_len': max_len,
            'placement_attempts': 400,  # Balance between attempts and quality
            'overlap_min': 0,
            'overlap_max': 3,  # Allow some overlap but not too much
            'density_target': 0.45,  # Good density but not overcrowded
            'min_words_required': max(3, word_count * 0.7)
        }

# ------------------------------------------------------------
# Word Filtering
# ------------------------------------------------------------
def filter_words(words: List[str], min_len: int, max_len: int, target_count: int) -> List[str]:
    """Filter words by length and quality."""
    # Filter by length and alphabetic
    filtered = [w.upper() for w in words
                if min_len <= len(w) <= max_len
                and w.isalpha()
                and len(w) >= 3]

    # Remove duplicates
    seen = set()
    unique_words = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)

    # Sort by length (mix it up for better variety)
    unique_words.sort(key=lambda x: (len(x), random.random()))

    # Ensure we have a good mix of lengths
    result = []
    length_buckets = defaultdict(list)

    for word in unique_words:
        length_buckets[len(word)].append(word)

    # Distribute word lengths evenly
    while len(result) < target_count * 3 and any(length_buckets.values()):
        for length in sorted(length_buckets.keys()):
            if length_buckets[length]:
                result.append(length_buckets[length].pop(0))
                if len(result) >= target_count * 3:
                    break

    return result[:target_count * 3]


# ------------------------------------------------------------
# Placement Helpers
# ------------------------------------------------------------
def calculate_word_score(grid: List[List[str]], word: str, r: int, c: int,
                         dy: int, dx: int, size: int, params: Dict) -> Tuple[bool, int]:
    """Calculate a score for word placement (higher is better)."""
    # Check bounds
    end_r = r + (len(word) - 1) * dy
    end_c = c + (len(word) - 1) * dx
    if not (0 <= end_r < size and 0 <= end_c < size):
        return False, 0

    score = 0
    overlaps = 0

    # Check each cell
    for i, ch in enumerate(word):
        rr = r + i * dy
        cc = c + i * dx
        cell = grid[rr][cc]

        if cell == '':
            # Empty cell - check if adjacent to other letters (prefer connections)
            for dr, dc in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] != '':
                    score += 1  # Bonus for being near other letters
                    break
        elif cell == ch:
            # Correct overlap
            overlaps += 1
            score += 5  # Good overlap bonus
        else:
            # Wrong letter - can't place here
            return False, 0

    # Check overlap constraints
    if overlaps < params['overlap_min'] or overlaps > params['overlap_max']:
        return False, 0

    # Bonus for using different directions
    if (dy, dx) not in [(0, 1), (1, 0)]:  # Not horizontal or vertical
        score += 3

    # Bonus for word length variety
    if len(word) <= 5:
        score += 1  # Short words are easier to find

    return True, score


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


def calculate_grid_density(grid: List[List[str]]) -> float:
    """Calculate what percentage of grid cells contain word letters."""
    size = len(grid)
    word_cells = sum(1 for r in range(size) for c in range(size) if grid[r][c] != '')
    return word_cells / (size * size)


# ------------------------------------------------------------
# Puzzle Generator
# ------------------------------------------------------------
def generate_puzzle(theme: str, size: int, params: Dict,
                    available_words: List[str], puzzle_id: str) -> Optional[Dict]:
    """Generate a single word search puzzle."""
    grid = [['' for _ in range(size)] for _ in range(size)]
    dirs = params['directions']
    attempts = params['placement_attempts']

    # Choose base words with good length distribution
    unique_available = list(dict.fromkeys(w.upper() for w in available_words))

    if len(unique_available) < params['word_count']:
        logging.warning(f"Not enough words for {puzzle_id}: {len(unique_available)} < {params['word_count']}")
        return None

    # Select words trying to get a mix of lengths
    base_words = []
    length_groups = defaultdict(list)

    for word in unique_available:
        length_groups[len(word)].append(word)

    # Distribute word lengths
    for _ in range(params['word_count']):
        # Prefer shorter words for easy, longer for hard
        target_lengths = list(length_groups.keys())
        if params['difficulty_label'] == 'easy':
            target_lengths.sort()  # Shorter first
        elif params['difficulty_label'] == 'hard':
            target_lengths.sort(reverse=True)  # Longer first
        else:
            random.shuffle(target_lengths)  # Mixed for medium

        for length in target_lengths:
            if length_groups[length]:
                base_words.append(length_groups[length].pop(0))
                break

    placed_original_words = []
    solution = []
    placed_positions = []  # Track where words are placed
    backwards_target = int(params['word_count'] * params['backwards_ratio'])
    backwards_used = 0

    # Try to place each word with intelligent positioning
    for word_idx, word in enumerate(base_words):
        original_word = word
        placed_word = word

        # Possibly reverse the word
        if backwards_used < backwards_target and random.random() < params['backwards_ratio']:
            placed_word = word[::-1]
            backwards_used += 1

        placed = False
        best_position = None
        best_score = -1

        # Try multiple positions and pick the best
        for attempt in range(attempts):
            dy, dx = random.choice(dirs)
            r = random.randint(0, size - 1)
            c = random.randint(0, size - 1)

            if can_place_word(grid, placed_word, r, c, dy, dx, size):
                can_place, score = calculate_word_score(grid, placed_word, r, c, dy, dx, size, params)
                if can_place and score > best_score:
                    best_score = score
                    best_position = (r, c, dy, dx)

        # Place at best position found
        if best_position:
            r, c, dy, dx = best_position
            place_word(grid, placed_word, r, c, dy, dx)
            pos = r * size + c
            dir_idx = ALL_DIRECTIONS.index((dy, dx))

            solution.append(f"{pos};{dir_idx};{len(original_word)};{original_word}")
            placed_original_words.append(original_word)
            placed = True

            # Track positions for this word
            word_positions = []
            for i in range(len(placed_word)):
                rr = r + i * dy
                cc = c + i * dx
                word_positions.append((rr, cc))
            placed_positions.append(word_positions)

        if not placed:
            logging.debug(f"Could not place word '{word}' in {puzzle_id}")

    # Check if we placed enough words
    if len(placed_original_words) < max(3, params['word_count'] * 0.7):
        logging.warning(f"Only placed {len(placed_original_words)} words in {puzzle_id}")
        return None

    # Check grid density
    density = calculate_grid_density(grid)
    if density < params['density_target'] * 0.7:
        logging.debug(f"Grid density too low: {density:.2f} in {puzzle_id}")
        # Still accept, but note it

    # Check for accidental word creation (false positives)
    false_positives = check_false_positives(grid, placed_positions, placed_original_words)
    if false_positives > 3:  # Allow a few accidental words
        logging.debug(f"Found {false_positives} false positives in {puzzle_id}")

    # Fill remaining empty cells with random letters
    for r in range(size):
        for c in range(size):
            if not grid[r][c]:
                # Avoid creating common short words
                letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

                # Check if this creates a 3-letter word accidentally
                if size >= 3:
                    # Check horizontal
                    if c >= 2 and grid[r][c - 2] != '' and grid[r][c - 1] != '':
                        potential_word = grid[r][c - 2] + grid[r][c - 1] + letter
                        if is_common_word(potential_word):
                            letter = random.choice('QXZJKV')  # Use less common letters

                grid[r][c] = letter

    # Flatten grid to string
    flat_grid = ''.join(''.join(row) for row in grid)

    # Final validation
    unique_placed = list(dict.fromkeys(w.upper() for w in placed_original_words))
    if len(unique_placed) != len(placed_original_words):
        logging.warning(f"Duplicate found in {puzzle_id}, deduplicating...")
    placed_original_words = unique_placed

    # Verify all placed words can be found in the grid
    if not verify_solution(grid, solution, size):
        logging.warning(f"Solution verification failed for {puzzle_id}")
        return None

    return {
        'id': puzzle_id,
        'theme': theme,
        'grid': flat_grid,
        'solution': ','.join(solution),
        'gridSize': size,
        'difficulty': params['difficulty_label'],
        'wordCount': len(placed_original_words),
        'wordlist': sorted(placed_original_words, key=lambda w: (len(w), w.lower()))
    }


def check_false_positives(grid: List[List[str]], placed_positions: List[List[Tuple[int, int]]],
                          placed_words: List[str]) -> int:
    """Check for accidentally created words not in the word list."""
    size = len(grid)
    false_positives = 0

    # Simple check: look for common 3-4 letter words
    for r in range(size):
        for c in range(size):
            # Check horizontal
            if c <= size - 3:
                word = grid[r][c] + grid[r][c + 1] + grid[r][c + 2]
                if is_common_word(word) and word not in placed_words:
                    false_positives += 1

            # Check vertical
            if r <= size - 3:
                word = grid[r][c] + grid[r + 1][c] + grid[r + 2][c]
                if is_common_word(word) and word not in placed_words:
                    false_positives += 1

    return false_positives


def is_common_word(word: str) -> bool:
    """Check if a 3-letter word is common."""
    common_3letter = {
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'ANY', 'CAN',
        'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM',
        'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO',
        'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'
    }
    return word in common_3letter


def verify_solution(grid: List[List[str]], solution: List[str], size: int) -> bool:
    """Verify that all words in the solution can be found in the grid."""
    try:
        for sol in solution:
            parts = sol.split(';')
            if len(parts) < 4:
                return False

            pos, dir_idx, length, word = parts
            pos = int(pos)
            dir_idx = int(dir_idx)
            length = int(length)

            if dir_idx >= len(ALL_DIRECTIONS):
                return False

            dy, dx = ALL_DIRECTIONS[dir_idx]
            start_row = pos // size
            start_col = pos % size

            # Check word in grid
            reconstructed = ''
            for i in range(length):
                r = start_row + i * dy
                c = start_col + i * dx
                if not (0 <= r < size and 0 <= c < size):
                    return False
                reconstructed += grid[r][c]

            if reconstructed != word and reconstructed != word[::-1]:
                return False

        return True
    except (ValueError, IndexError):
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
            max_failed_attempts = 100

            logging.info(f"Generating {PUZZLES_PER_COMBO} puzzles for size={size}, difficulty={diff}")

            # Keep track of which themes we've used in this batch
            used_themes = []
            available_themes = []
            for theme in themes:
                # if theme in used_themes:
                #    continue

                words = theme_cache.get(theme, [])
                if not words or len(words) < 30:  # Need enough words
                    continue

                usable = filter_words(words, params['min_len'], params['max_len'], params['word_count'])
                if len(usable) >= params['word_count'] * 2:  # Need plenty of options
                    available_themes.append((theme, usable))

            while puzzles_made < PUZZLES_PER_COMBO and failed_attempts < max_failed_attempts:
                # Find themes with enough words that we haven't used yet
                # Randomly select a theme from available ones
                theme, usable_words = random.choice(available_themes)

                # Generate one puzzle from this theme
                puzzle_id = f"{size}-{diff}-{file_counter}"

                try:
                    puzzle = generate_puzzle(theme, size, params, usable_words, puzzle_id)

                    if not puzzle:
                        failed_attempts += 1
                        if failed_attempts % 10 == 0:
                            logging.warning(f"{failed_attempts} failed attempts for {size}-{diff}")
                        continue

                    # Validate puzzle
                    if len(puzzle['grid']) != size * size:
                        logging.error(f"Invalid grid size in {puzzle_id}")
                        failed_attempts += 1
                        continue

                    # Check for duplicate words in wordlist
                    wordlist = puzzle['wordlist']
                    if len(wordlist) != len(set(w.upper() for w in wordlist)):
                        logging.error(f"Duplicate words in {puzzle_id}")
                        failed_attempts += 1
                        continue

                    # Check word count
                    if len(wordlist) < max(3, params['word_count'] * 0.7):
                        logging.warning(f"Too few words in {puzzle_id}: {len(wordlist)}")
                        failed_attempts += 1
                        continue

                    # Save puzzle
                    output_file = folder / f"{file_counter}.json"
                    with open(output_file, 'w') as f:
                        json.dump(puzzle, f, separators=(',', ':'))

                    puzzles_made += 1
                    file_counter += 1
                    failed_attempts = 0  # Reset counter on success

                    if puzzles_made % 100 == 0:
                        logging.info(f"  Generated {puzzles_made}/{PUZZLES_PER_COMBO} puzzles")

                except Exception as e:
                    logging.error(f"Error generating puzzle {puzzle_id}: {e}")
                    failed_attempts += 1
                    continue

            if failed_attempts >= max_failed_attempts:
                logging.error(f"Too many failed attempts for {size}-{diff}, stopping")

            logging.info(
                f"Completed: size={size}, difficulty={diff} - {puzzles_made} puzzles)")

    logging.info("Generation complete!")


if __name__ == "__main__":
    main()