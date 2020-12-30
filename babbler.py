from collections import defaultdict
from random import choices


def distribution():
    result = defaultdict(int)
    for i in range(26):
        result[chr(i + ord('a'))] = 0.0001
    result[' '] = .0001
    return result
 

class Babbler:
 def __init__(self, n, filename):
   self.counts = defaultdict(distribution)
   self.a = n
   with open(filename) as file:
     self.text = file.read().lower().replace('\n', ' ')
   for letter in range(0,len(self.text) - n , 1):
      self.counts[self.text[letter:letter + n]][self.text[letter + n]] += 1
 def babble(self, s, length ):
    a = self.a
    sentence = s[:a]
    for i in range(0,length, a):
      sentence += choices(list(self.counts[sentence[ -a:]].keys()), weights = list(self.counts[sentence[-a : ]].values()))[0]
    print(sentence)
