# ============================================================
# Author   : Mobeen Anwar
# Degree   : BSc Computer Science (81%) — Pakistan
# Currently: MSc High Integrity Systems — Frankfurt, Germany
# Topic    : Data Mining — Exercise 1: Python Basics
# GitHub   : github.com/mobeenanwardev-ui
# ============================================================

# ------------------------------------------------------------
# WHAT I LEARNED IN THIS EXERCISE:
# 1. How to use Shell Commands in Jupyter
# 2. How to import packages (numpy, pandas)
# 3. How to access documentation in Python
# 4. Mutable vs Immutable Objects
# 5. List Comprehension
# 6. Assert Statements
# 7. Functions — add_values, cumulative_sum, is_palindrome,
#                find_the_a
# ------------------------------------------------------------


# ============================================================
# SECTION 1: Shell Commands
# ============================================================
# In Jupyter Notebook you can talk directly to your computer.
# %pwd  → tells you which folder you are currently in
# !ls   → shows all files in that folder
# These are not Python — they talk to the computer's terminal.


# ============================================================
# SECTION 2: Importing Packages
# ============================================================
# Packages are like toolboxes.
# numpy  → super fast math on numbers and arrays
# pandas → working with tables of data (like Excel in Python)

import numpy as np
import pandas as pd


# ============================================================
# SECTION 3: Accessing Documentation
# ============================================================
# Python has built-in help tools so you never have to Google:
#   pd.DataFrame?   → quick description of DataFrame
#   np.mean??       → shows the actual source code of mean
#   object. + Tab   → lists all methods available


# ============================================================
# SECTION 4: Mutable vs Immutable
# ============================================================

# IMMUTABLE = cannot be changed after creation (written in stone)
# Examples: tuple, string, int, float, bool

# tuple1 = (0, 1, 2, 3)
# tuple1[0] = 4       ← this gives an ERROR — tuples are locked!

# MUTABLE = can be changed freely (written on whiteboard)
# Examples: list, dictionary, set

my_list = [1, 2, 3]
my_list.append(4)       # adds 4 to the end    → [1, 2, 3, 4]
my_list.insert(1, 5)    # inserts 5 at index 1 → [1, 5, 2, 3, 4]
my_list.remove(2)       # removes value 2       → [1, 5, 3, 4]
popped = my_list.pop(0) # removes index 0       → [5, 3, 4]

print("List after operations:", my_list)
print("Popped element:", popped)

# ⚠️ IMPORTANT: new_list = my_list is NOT a copy!
# Both variables point to the SAME data in memory.
# Change one → the other changes too!


# ============================================================
# SECTION 5: List Comprehension
# ============================================================
# A shortcut to create lists in one line instead of a for loop.

# Normal way:
evens_normal = []
for i in range(10):
    if i % 2 == 0:
        evens_normal.append(i)

# List comprehension way (same result, one line):
evens_short = [i for i in range(10) if i % 2 == 0]

print("Even numbers:", evens_short)  # [0, 2, 4, 6, 8]

# Odd numbers:
odds = [i for i in range(10) if i % 2 != 0]
print("Odd numbers:", odds)          # [1, 3, 5, 7, 9]

# Fun example — Star triangle using list comprehension:
[print('* ' * i) for i in range(1, 6)]


# ============================================================
# SECTION 6: Assert Statements
# ============================================================
# Assert is like a security guard.
# If condition is TRUE  → silent, no problem ✅
# If condition is FALSE → AssertionError, program stops ❌
# Used to TEST your own functions during development.

assert (1 + 5) == 6    # ✅ passes silently
# assert (1 + 6) == 8  # ❌ would crash — commented out on purpose


# ============================================================
# SECTION 7: Functions
# ============================================================

# ------------------------------------------------------------
# Function 1: add_values
# Takes two numbers, returns their sum.
# Simple but important — foundation of all functions.
# ------------------------------------------------------------
def add_values(a, b):
    """Calculate the sum a + b"""
    return a + b

assert add_values(1, 3) == 4
assert add_values(-4, 2) == -2
print("add_values works:", add_values(10, 5))  # 15


# ------------------------------------------------------------
# Function 2: cumulative_sum
# Returns the sum of all numbers from 1 to n.
# Example: cumulative_sum(5) = 1+2+3+4+5 = 15
# range(1, n+1) — we write n+1 because range stops before the end!
# ------------------------------------------------------------
def cumulative_sum(n):
    """Calculate the sum 1 + 2 + 3 + ... + n"""
    return sum(range(1, n + 1))

assert cumulative_sum(1) == 1
assert cumulative_sum(5) == 15
assert cumulative_sum(100) == 5050
print("cumulative_sum(100):", cumulative_sum(100))  # 5050


# ------------------------------------------------------------
# Function 3: is_palindrome
# Returns True if the word reads the same forwards and backwards.
# Examples: "racecar" ✅  "hello" ❌
# [::-1] is Python's trick to reverse a string — read right to left!
# ------------------------------------------------------------
def is_palindrome(s):
    """Returns True when s does not change when being reversed"""
    return s == s[::-1]

assert is_palindrome("racecar") == True
assert is_palindrome("foo") == False
assert is_palindrome("abba") == True
print("is_palindrome('racecar'):", is_palindrome("racecar"))  # True


# ------------------------------------------------------------
# Function 4: find_the_a
# Returns the position (index) of the first "a" in a string.
# Returns False if there is no "a".
# Remember: Python counts from 0! (H=0, a=1, l=2, l=3, o=4)
# ------------------------------------------------------------
def find_the_a(s):
    """Returns the position of the first 'a' in s or False if none exists"""
    if "a" in s:
        return s.index("a")
    return False

assert find_the_a("Hallo Welt!") == 1
assert find_the_a("edcba") == 4
assert find_the_a("Hello world!") == False
assert find_the_a("") == False
print("find_the_a('Hallo'):", find_the_a("Hallo"))  # 1


# ============================================================
# ALL TESTS PASSED ✅
# ============================================================
print("\n✅ All functions implemented and tested successfully!")
print("🚀 Exercise 1 Complete — Mobeen Anwar")
