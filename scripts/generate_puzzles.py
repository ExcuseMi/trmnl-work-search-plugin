#!/usr/bin/env python3
"""
Generate word search puzzles and save as JSON files.
Organized by: data/{size}/{difficulty}/{id}.json starting at 1.json

Professional difficulty levels based on research:
- Easy: Fewer words, only forward directions (right/down)
- Medium: More words, 4 directions (no backwards)
- Hard: Most words, all 8 directions
"""

import requests
import random
import json
import time
from pathlib import Path

# Configuration
THEMES = [
    "beach", "space", "ocean", "forest", "mountain", "desert", "city", "music",
    "sports", "food", "animals", "weather", "travel", "technology", "art", "science",
    "garden", "winter", "summer", "spring", "autumn", "coffee", "book", "movie",
    "fitness", "cooking", "adventure", "nature", "holiday", "festival", "astronomy",
    "history", "architecture", "photography", "health", "fashion", "education",
    "business", "mythology", "fantasy", "friendship", "family", "home", "childhood",
    "nostalgia", "meditation", "mindfulness", "crafts", "diy", "vintage", "futurism",
    "minimalism", "luxury", "sustainability", "farming", "camping", "roadtrip",
    "nightlife", "sunrise", "sunset", "rain", "snow", "cat", "dog", "bird", "flower",
    "tree", "river", "lake", "island", "cave", "volcano", "concert", "theater",
    "painting", "sculpture", "poetry", "writing", "gaming", "podcast", "yoga",
    "cycling", "hiking", "surfing", "skateboarding", "baking", "vegan", "streetfood",
    "wine", "tea", "chocolate", "comedy", "romance", "mystery", "horror"
]

GRID_SIZES = [10, 12, 15]
DIFFICULTIES = ["easy", "medium", "hard"]
PUZZLES_PER_THEME = 40

OUTPUT_DIR = Path("data")
ALL_DIRECTIONS = [[0, 1], [1, 0], [1, 1], [-1, 1], [0, -1], [-1, 0], [-1, -1], [1, -1]]
FALLBACK_WORDS = ['PUZZLE', 'SEARCH', 'FIND', 'WORD', 'GAME', 'FUN', 'BRAIN', 'SOLVE', 'GRID', 'LETTERS']


def get_difficulty_params(difficulty, grid_size):
    """Return parameters based on professional puzzle standards"""
    if difficulty == "easy":
        # Easy: Only forward directions (right and down) - 50-60% grid density
        allowed_directions = [[0, 1], [1, 0]]  # Right, Down
        word_count = max(6, int(grid_size * 0.6))  # 60% of grid size
        min_len = 4
        max_len = min(8, grid_size - 1)
        grid_density = 0.6

    elif difficulty == "medium":
        # Medium: 4 directions (no backwards) - 70-80% grid density
        allowed_directions = [[0, 1], [1, 0], [1, 1], [-1, 1]]  # Right, Down, Diagonal down-right, Diagonal up-right
        word_count = max(8, int(grid_size * 0.8))  # 80% of grid size
        min_len = 4
        max_len = min(10, grid_size - 1)
        grid_density = 0.8

    else:  # hard
        # Hard: All 8 directions - 90-100% grid density
        allowed_directions = ALL_DIRECTIONS  # All 8 directions including backwards
        word_count = max(10, int(grid_size * 1.0))  # 100% of grid size
        min_len = 3  # Allow shorter words for extra challenge
        max_len = min(12, grid_size - 1)
        grid_density = 1.0

    return {
        'word_count': word_count,
        'directions': allowed_directions,
        'min_len': min_len,
        'max_len': max_len,
        'grid_density': grid_density
    }


def fetch_theme_words(theme):
    try:
        url = f"https://api.datamuse.com/words?rel_trg={theme}&max=100"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return [w['word'].upper() for w in response.json()]
    except:
        return []


def filter_words(words, min_len, max_len, count):
    filtered = [w for w in words if min_len <= len(w) <= max_len and w.isalpha()]
    selected = []
    for word in filtered:
        if not any(word != other and (word in other or other in word) for other in filtered):
            if word not in selected:
                selected.append(word)
        if len(selected) >= count * 5:  # Get 5x more words than needed
            break
    return selected


def can_place_word(grid, word, row, col, dy, dx, grid_size):
    end_row, end_col = row + (len(word) - 1) * dy, col + (len(word) - 1) * dx
    if not (0 <= end_row < grid_size and 0 <= end_col < grid_size):
        return False
    for i, char in enumerate(word):
        r, c = row + i * dy, col + i * dx
        if grid[r][c] and grid[r][c] != char:
            return False
    return True


def place_word(grid, word, row, col, dy, dx):
    for i, char in enumerate(word):
        grid[row + i * dy][col + i * dx] = char


def generate_puzzle(theme, grid_size, difficulty_params, available_words, puzzle_id):
    grid = [['' for _ in range(grid_size)] for _ in range(grid_size)]
    word_count = difficulty_params['word_count']
    allowed_directions = difficulty_params['directions']

    puzzle_words = random.sample(available_words, min(word_count, len(available_words)))
    solution = []
    fallback_index = 0
    wordlist = []  # Store the actual words placed

    for word in puzzle_words:
        placed = False
        for _ in range(200):  # Increased attempts for more challenging placements
            row, col = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            dy, dx = random.choice(allowed_directions)
            if can_place_word(grid, word, row, col, dy, dx, grid_size):
                place_word(grid, word, row, col, dy, dx)
                solution.append({'start': [row, col], 'direction': [dy, dx], 'length': len(word)})
                wordlist.append(word)  # Add to wordlist
                placed = True
                break

        if not placed and fallback_index < len(FALLBACK_WORDS):
            word = FALLBACK_WORDS[fallback_index]
            fallback_index += 1
            for _ in range(100):
                row, col = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
                dy, dx = random.choice(allowed_directions)
                if can_place_word(grid, word, row, col, dy, dx, grid_size):
                    place_word(grid, word, row, col, dy, dx)
                    solution.append({'start': [row, col], 'direction': [dy, dx], 'length': len(word)})
                    wordlist.append(word)  # Add to wordlist
                    break

    # Fill remaining cells with random letters
    for r in range(grid_size):
        for c in range(grid_size):
            if not grid[r][c]:
                grid[r][c] = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    grid_str = ''.join(''.join(row) for row in grid)

    # Create direction mapping for encoding
    dir_map = {}
    for idx, (dy, dx) in enumerate(ALL_DIRECTIONS):
        dir_map[(dy, dx)] = idx

    encoded = []
    for s in solution:
        pos = s['start'][0] * grid_size + s['start'][1]
        dir_idx = dir_map[tuple(s['direction'])]
        encoded.append(f"{pos};{dir_idx};{s['length']}")

    return {
        'id': puzzle_id,  # Add the ID
        'theme': theme,
        'grid': grid_str,
        'solution': ','.join(encoded),
        'gridSize': grid_size,
        'difficulty': difficulty_params.get('difficulty_label', ''),
        'wordCount': len(solution),
        'wordlist': wordlist  # Add the wordlist
    }


def main():
    print(f"Generating {PUZZLES_PER_THEME} puzzles per theme...")
    print("Difficulty levels based on professional standards:")
    print("  Easy: Fewer words, only forward directions (right/down)")
    print("  Medium: More words, 4 directions (no backwards)")
    print("  Hard: Most words, all 8 directions")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # New index structure with per-folder indexing
    index = {
        str(size): {
            diff: {
                'start': 1,  # All folders start at 1.json
                'count': 0,
                'themes': {}
            } for diff in DIFFICULTIES
        } for size in GRID_SIZES
    }

    total_puzzles, total_bytes = 0, 0

    # Pre-fetch words
    print("\nFetching words from API...")
    theme_cache = {}
    for i, theme in enumerate(THEMES):
        print(f"  [{i + 1}/{len(THEMES)}] {theme}...", end='')
        words = fetch_theme_words(theme)
        if words:
            theme_cache[theme] = words
            print(f" ✓ {len(words)}")
        else:
            print(" ✗")
        time.sleep(0.3)

    print(f"\nFetched {len(theme_cache)}/{len(THEMES)} themes\n")

    # Generate puzzles
    for grid_size in GRID_SIZES:
        for difficulty in DIFFICULTIES:
            # Get difficulty parameters
            params = get_difficulty_params(difficulty, grid_size)
            params['difficulty_label'] = difficulty

            print(f"\nGenerating {grid_size}×{grid_size} {difficulty.upper()}")
            print(f"  Words: {params['word_count']}, Directions: {len(params['directions'])}, "
                  f"Length: {params['min_len']}-{params['max_len']} letters")

            puzzle_dir = OUTPUT_DIR / str(grid_size) / difficulty
            puzzle_dir.mkdir(parents=True, exist_ok=True)

            # Reset file counter for this folder
            file_counter = 1

            for theme in theme_cache:
                # Filter words with difficulty-specific length constraints
                filtered = filter_words(
                    theme_cache[theme],
                    params['min_len'],
                    params['max_len'],
                    params['word_count']
                )

                if len(filtered) < params['word_count']:
                    continue

                theme_puzzle_count = 0
                theme_start_id = file_counter

                for _ in range(PUZZLES_PER_THEME):
                    # Create puzzle ID in format: "size-difficulty-filenumber"
                    puzzle_id = f"{grid_size}-{difficulty}-{file_counter}"

                    puzzle = generate_puzzle(theme, grid_size, params, filtered, puzzle_id)
                    puzzle_file = puzzle_dir / f"{file_counter}.json"

                    with open(puzzle_file, 'w') as f:
                        json.dump(puzzle, f, separators=(',', ':'))

                    total_bytes += puzzle_file.stat().st_size
                    total_puzzles += 1
                    theme_puzzle_count += 1
                    file_counter += 1

                # Record theme info if we generated puzzles for it
                if theme_puzzle_count > 0:
                    index[str(grid_size)][difficulty]['themes'][theme] = {
                        'start': theme_start_id,
                        'count': theme_puzzle_count
                    }

            # Update folder metadata
            index[str(grid_size)][difficulty]['count'] = file_counter - 1
            print(f"  Generated {file_counter - 1} puzzles (files: 1.json to {file_counter - 1}.json)")

    # Save the index file
    with open(OUTPUT_DIR / 'index.json', 'w') as f:
        json.dump(index, f, separators=(',', ':'))

    print(f"\n{'=' * 60}")
    print(f"Complete! {total_puzzles} puzzles, {total_bytes / 1024 / 1024:.2f} MB")
    print(f"Average: {total_bytes / total_puzzles:.0f} bytes/puzzle")
    print(f"Files numbered from 1.json in each folder")

    # Summary of difficulty settings
    print(f"\nDifficulty Settings Summary:")
    for size in GRID_SIZES:
        print(f"\n{size}×{size} grid:")
        for diff in DIFFICULTIES:
            params = get_difficulty_params(diff, size)
            print(f"  {diff.upper():6s}: {params['word_count']:2d} words, "
                  f"{len(params['directions']):2d} directions, "
                  f"{params['min_len']}-{params['max_len']} letters")

    print(f"{'=' * 60}")

    # Create a sample puzzle to show the format
    print(f"\nSample puzzle format:")
    sample_puzzle = {
        "id": "10-easy-1",
        "theme": "space",
        "grid": "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "solution": "12;0;7,53;3;6,11;1;6",
        "gridSize": 10,
        "difficulty": "easy",
        "wordCount": 3,
        "wordlist": ["STAR", "PLANET", "MOON"]
    }
    print(json.dumps(sample_puzzle, indent=2))


if __name__ == "__main__":
    main()