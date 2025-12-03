#!/usr/bin/env python3
"""
Sort wordlists AND solutions together to maintain color-word matching.
Sorts by: length ascending, then alphabetically case-insensitive.
"""

import json
import os
from pathlib import Path

# Direction mapping (same as in your generator)
ALL_DIRECTIONS = [[0, 1], [1, 0], [1, 1], [-1, 1], [0, -1], [-1, 0], [-1, -1], [1, -1]]


def extract_words_from_solution(grid_str, solution_str, grid_size):
    """Extract words from solution string to match with wordlist"""
    if not solution_str:
        return []

    dir_map = {tuple(dir): idx for idx, dir in enumerate(ALL_DIRECTIONS)}
    reverse_dir_map = {idx: dir for idx, dir in enumerate(ALL_DIRECTIONS)}

    solution_parts = solution_str.split(',')
    words = []

    for part in solution_parts:
        if not part:
            continue
        pos_str, dir_str, length_str = part.split(';')
        pos = int(pos_str)
        dir_idx = int(dir_str)
        length = int(length_str)

        start_row = pos // grid_size
        start_col = pos % grid_size
        dy, dx = reverse_dir_map[dir_idx]

        # Build the word
        word = ''
        for i in range(length):
            row = start_row + i * dy
            col = start_col + i * dx
            grid_index = row * grid_size + col
            word += grid_str[grid_index]

        words.append(word)

    return words


def sort_puzzle_together(puzzle_data):
    """Sort wordlist and solution together maintaining correspondence"""
    if 'wordlist' not in puzzle_data or 'solution' not in puzzle_data:
        return puzzle_data

    grid_str = puzzle_data['grid']
    grid_size = puzzle_data['gridSize']
    solution_str = puzzle_data['solution']

    # Extract current words from solution to verify
    solution_words = extract_words_from_solution(grid_str, solution_str, grid_size)

    # Verify wordlist matches solution words
    if len(puzzle_data['wordlist']) != len(solution_words):
        print(f"⚠️  Warning: wordlist length doesn't match solution length")
        return puzzle_data

    # Create list of (word_from_wordlist, word_from_solution, solution_part, index)
    combined = []
    solution_parts = solution_str.split(',')

    for idx, (wordlist_word, solution_word, solution_part) in enumerate(
            zip(puzzle_data['wordlist'], solution_words, solution_parts)
    ):
        # Verify they match (they should!)
        if wordlist_word != solution_word:
            print(f"⚠️  Warning: Mismatch at index {idx}: '{wordlist_word}' vs '{solution_word}'")

        combined.append({
            'wordlist_word': wordlist_word,
            'solution_word': solution_word,
            'solution_part': solution_part,
            'original_index': idx
        })

    # Sort by length, then alphabetically case-insensitive
    combined_sorted = sorted(
        combined,
        key=lambda x: (len(x['wordlist_word']), x['wordlist_word'].lower())
    )

    # Extract sorted components
    sorted_wordlist = [item['wordlist_word'] for item in combined_sorted]
    sorted_solution_parts = [item['solution_part'] for item in combined_sorted]

    # Update puzzle data
    puzzle_data['wordlist'] = sorted_wordlist
    puzzle_data['solution'] = ','.join(sorted_solution_parts)

    return puzzle_data


def process_puzzle_file(file_path):
    """Process a single puzzle file"""
    try:
        with open(file_path, 'r') as f:
            puzzle_data = json.load(f)

        original_wordlist = puzzle_data.get('wordlist', [])[:]
        original_solution = puzzle_data.get('solution', '')

        updated = sort_puzzle_together(puzzle_data)

        # Check if anything changed
        if (original_wordlist != updated.get('wordlist', []) or
                original_solution != updated.get('solution', '')):
            with open(file_path, 'w') as f:
                json.dump(updated, f, separators=(',', ':'))
            return True

        return False

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False


def verify_sorting(puzzle_data):
    """Verify that sorting is correct"""
    grid_str = puzzle_data['grid']
    grid_size = puzzle_data['gridSize']
    solution_str = puzzle_data['solution']
    wordlist = puzzle_data['wordlist']

    # Extract words from newly sorted solution
    extracted_words = extract_words_from_solution(grid_str, solution_str, grid_size)

    # Verify they match
    for i, (wl_word, ex_word) in enumerate(zip(wordlist, extracted_words)):
        if wl_word != ex_word:
            return False, f"Mismatch at index {i}: '{wl_word}' vs '{ex_word}'"

    # Verify sorted order
    for i in range(len(wordlist) - 1):
        current = wordlist[i]
        next_word = wordlist[i + 1]

        # Check length ordering
        if len(current) > len(next_word):
            return False, f"Length order wrong at index {i}: '{current}' ({len(current)}) > '{next_word}' ({len(next_word)})"

        # If same length, check alphabetical
        if len(current) == len(next_word) and current.lower() > next_word.lower():
            return False, f"Alphabetical order wrong at index {i}: '{current}' > '{next_word}'"

    return True, "OK"


def main():
    base_dirs = ['data', 'mobile']

    total_files = 0
    total_updated = 0
    total_errors = 0
    verification_errors = []

    for base_dir in base_dirs:
        if not Path(base_dir).exists():
            print(f"⚠️  Directory '{base_dir}' not found, skipping...")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing {base_dir}/ directory")
        print(f"{'=' * 60}")

        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = Path(root) / file
                    total_files += 1

                    if total_files % 500 == 0:
                        print(f"  Processed {total_files} files...")

                    try:
                        updated = process_puzzle_file(file_path)
                        if updated:
                            total_updated += 1

                            # Verify the sorting
                            with open(file_path, 'r') as f:
                                puzzle = json.load(f)
                            ok, msg = verify_sorting(puzzle)
                            if not ok:
                                verification_errors.append(f"{file_path}: {msg}")

                    except Exception as e:
                        total_errors += 1
                        print(f"❌ Error with {file_path}: {e}")

    print(f"\n{'=' * 60}")
    print("SORTING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total files processed: {total_files}")
    print(f"Files updated: {total_updated}")
    print(f"Files with errors: {total_errors}")

    if verification_errors:
        print(f"\n⚠️  Verification errors ({len(verification_errors)}):")
        for error in verification_errors[:5]:  # Show first 5
            print(f"  {error}")
        if len(verification_errors) > 5:
            print(f"  ... and {len(verification_errors) - 5} more")

    # Show example of sorting
    print(f"\nExample of sorting logic:")
    test_data = {
        'wordlist': ["ZEBRA", "apple", "Cat", "dog", "ELEPHANT", "banana", "ant"],
        'solution': "0;0;5,1;0;5,2;0;3,3;0;3,4;0;8,5;0;6,6;0;3",  # Fake solution for demo
    }

    print(f"Before: {test_data['wordlist']}")

    # Simulate sorting
    sorted_indices = sorted(
        range(len(test_data['wordlist'])),
        key=lambda i: (len(test_data['wordlist'][i]), test_data['wordlist'][i].lower())
    )

    sorted_words = [test_data['wordlist'][i] for i in sorted_indices]
    print(f"After:  {sorted_words}")

    print(f"\n✅ Wordlist and solution sorted together")
    print(f"✅ Index.html colors will match displayed words")
    print(f"✅ Order: Shortest first, then alphabetical (case-insensitive)")


if __name__ == "__main__":
    main()