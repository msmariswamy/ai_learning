import math
from collections import Counter

def distinct_permutations(word: str) -> int:
    """
    Compute the number of distinct permutations of `word`,
    accounting for repeated characters (case‑sensitive).
    """
    n = len(word)
    total = math.factorial(n)
    for cnt in Counter(word).values():
        total //= math.factorial(cnt)
    return total

if __name__ == "__main__":
    word = "ALGEbRA"
    print(distinct_permutations(word))