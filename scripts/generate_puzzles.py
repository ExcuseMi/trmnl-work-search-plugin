#!/usr/bin/env python3
"""
Generate word search puzzles and save as JSON files.
Organized by: data/{size}/{difficulty}/{id}.json starting at 1.json

40 puzzles per theme × ~80 successful themes × 3 sizes × 3 difficulties = ~28,800 puzzles (~7MB)
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
DIRECTIONS = [[0, 1], [1, 0], [1, 1], [-1, 1], [0, -1], [-1, 0], [-1, -1], [1, -1]]
FALLBACK_WORDS = ['PUZZLE', 'SEARCH', 'FIND', 'WORD', 'GAME', 'FUN', 'BRAIN', 'SOLVE', 'GRID', 'LETTERS']


def fetch_theme_words(theme):
    try:
        url = f"https://api.datamuse.com/words?rel_trg={theme}&max=100"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return [w['word'].upper() for w in response.json()]
    except:
        return []


def filter_words(words, grid_size, count):
    min_len, max_len = 4, min(10, grid_size - 1)
    filtered = [w for w in words if min_len <= len(w) <= max_len and w.isalpha()]
    selected = []
    for word in filtered:
        if not any(word != other and (word in other or other in word) for other in filtered):
            if word not in selected:
                selected.append(word)
        if len(selected) >= count * 5:
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


def generate_puzzle(theme, grid_size, word_count, available_words):
    grid = [['' for _ in range(grid_size)] for _ in range(grid_size)]
    puzzle_words = random.sample(available_words, min(word_count, len(available_words)))
    solution = []
    fallback_index = 0

    for word in puzzle_words:
        placed = False
        for _ in range(100):
            row, col = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            dy, dx = random.choice(DIRECTIONS)
            if can_place_word(grid, word, row, col, dy, dx, grid_size):
                place_word(grid, word, row, col, dy, dx)
                solution.append({'start': [row, col], 'direction': [dy, dx], 'length': len(word)})
                placed = True
                break

        if not placed and fallback_index < len(FALLBACK_WORDS):
            word = FALLBACK_WORDS[fallback_index]
            fallback_index += 1
            for _ in range(100):
                row, col = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
                dy, dx = random.choice(DIRECTIONS)
                if can_place_word(grid, word, row, col, dy, dx, grid_size):
                    place_word(grid, word, row, col, dy, dx)
                    solution.append({'start': [row, col], 'direction': [dy, dx], 'length': len(word)})
                    break

    for r in range(grid_size):
        for c in range(grid_size):
            if not grid[r][c]:
                grid[r][c] = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    grid_str = ''.join(''.join(row) for row in grid)
    dir_map = {(0, 1): 0, (1, 0): 1, (1, 1): 2, (-1, 1): 3, (0, -1): 4, (-1, 0): 5, (-1, -1): 6, (1, -1): 7}
    encoded = [f"{s['start'][0] * grid_size + s['start'][1]};{dir_map[tuple(s['direction'])]};{s['length']}" for s in
               solution]

    return {'theme': theme, 'grid': grid_str, 'solution': ','.join(encoded), 'gridSize': grid_size}


def main():
    print(f"Generating {PUZZLES_PER_THEME} puzzles per theme...")
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
            word_count = 6 if difficulty == "easy" else 8 if difficulty == "medium" else 10
            print(f"Generating {grid_size}×{grid_size} {difficulty} (word count: {word_count})")

            puzzle_dir = OUTPUT_DIR / str(grid_size) / difficulty
            puzzle_dir.mkdir(parents=True, exist_ok=True)

            # Reset file counter for this folder
            file_counter = 1

            for theme in theme_cache:
                filtered = filter_words(theme_cache[theme], grid_size, word_count)
                if len(filtered) < word_count:
                    continue

                theme_puzzle_count = 0
                theme_start_id = file_counter

                for _ in range(PUZZLES_PER_THEME):
                    puzzle = generate_puzzle(theme, grid_size, word_count, filtered)
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

    print(f"\n{'=' * 60}")
    print(f"Complete! {total_puzzles} puzzles, {total_bytes / 1024 / 1024:.2f} MB")
    print(f"Average: {total_bytes / total_puzzles:.0f} bytes/puzzle")
    print(f"Files numbered from 1.json in each folder")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()