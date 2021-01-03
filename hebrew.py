from bs4 import BeautifulSoup
import os
import numpy as np
import unicodedata


def read_bible():
    """
    Need to download bible in htm format from Machon Mamre:
    https://www.mechon-mamre.org/dlk.htm
    """
    txt = u''
    fr_ = 168  # discard first 200 chars of the first file,
    for i in range(1, 36):
        with open(os.path.join('k', f'k{i:02d}.htm'), encoding='windows-1255') as R:
            raw = R.read()
            soup = BeautifulSoup(raw, features='html.parser')
            txt += soup.find('body').get_text()[fr_:]
            fr_ = 0  # for all other files - take everything.
    txt = unicodedata.normalize("NFKD", txt)
    # remove rare characters (less than 100 occurrences)
    for c in set(txt):
        if txt.count(c) < 100:
            txt = txt.replace(c, '')
    return txt


def convert_utf8_to_tokens(txt):
    """
    convert the utf8 text into 0-n tokens and stores a dictionary for the mapping back

    returns:
    code
    dictionary
    """
    dictionary, idx, rev_idx, count = np.unique(list(txt), return_index=True, return_inverse=True, return_counts=True)
    print(f'\tReading text: got {len(rev_idx)} total chars, and {len(dictionary)} uniq.')
    return rev_idx, dictionary


def code_to_text(code, dictionary):
    return ''.join(dictionary[code])


def text_to_code(txt, dictionary):
    return (dictionary[None, :] == np.array(list(txt))[:, None]).argmax(axis=1)
