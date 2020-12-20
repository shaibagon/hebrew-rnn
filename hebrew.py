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
    for i in range(1, 36):
        with open(os.path.join('k', f'k{i:02d}.htm'), encoding='windows-1255') as R:
            raw = R.read()
            soup = BeautifulSoup(raw, features='html.parser')
            txt += soup.get_text()
    return unicodedata.normalize("NFKD", txt)


def convert_utf8_to_tokens(txt):
    """
    convert the utf8 text into 0-n tokens and stores a dictionary for the mapping back

    returns:
    code
    dictionary
    """
    u, idx, rev_idx, count = np.unique(list(txt), return_index=True, return_inverse=True, return_counts=True)
    return rev_idx, u


def code_to_text(code, dictionary):
    return ''.join(dictionary[code])


def text_to_code(txt, dictionary):
    return (dictionary[None, :] == np.array(list(txt))[:, None]).argmax(axis=1)
