#!/usr/bin/env python3
# filename: permutation_count.py
import math
from collections import Counter

def distinct_permutations(word: str) -> int:
    """Return the number of distinct permutations of `word`."""
    n = len(word)
    total = math.factorial(n)               # n!
    for cnt in Counter(word).values():      # divide by each repeated count!
        total //= math.factorial(cnt)
    return total

if __name__ == "__main__":
    word = "ALGEbRA"
    result = distinct_permutations(word)
    print(f"The number of distinct permutations of '{word}' is: {result}")