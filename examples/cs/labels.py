from pprint import pprint
from cs_lib import cs_lib

if __name__ == "__main__":
    text = "aabbcc hello world!"
    counts = cs_lib.count_characters(text)
    pprint(counts)
