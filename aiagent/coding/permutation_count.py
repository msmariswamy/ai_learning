# filename: permutation_count.py
import math
from collections import Counter

# The word to analyze
word = "ALGEbRA"

# Count how many times each character appears
char_counts = Counter(word)

# Total length of the word
n = len(word)

# Start with n!
total_permutations = math.factorial(n)

# Divide by the factorial of each character's count
for cnt in char_counts.values():
    total_permutations //= math.factorial(cnt)

# Output the final count
print(total_permutations)