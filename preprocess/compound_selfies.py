'''
SELFIES, a chemical language for encoding molecules, alternative to SMILES
https://github.com/aspuru-guzik-group/selfies
'''
import selfies as sf

import re


def shorten(s):
    # 使用正则表达式替换字符串中的多个子字符串
    s = re.sub(r"Branch", "$", s)
    s = re.sub(r"Ring", "%", s)
    s = re.sub(r"\]", "", s)
    s = re.sub(r"\[", "", s)
    return s


def encode(smiles, character=[]):
    if isinstance(smiles, str):
        return sf.encoder(smiles, strict=False)
    else:
        lookup = {v: i for i, v in enumerate(character)}
        ret = []
        for i in smiles:
            try:
                i=i.strip()
                s = sf.encoder(i, strict=False)
            except Exception as e:
                print(i)
                raise e
                continue
            k = []
            # _ added for 1.5 bond, $=branch, %=ring,eliminate ] for short,
            # replace [ with |
            s = shorten(s)
            for char in s:
                assert char in lookup, f"{char} not in lookup, sentence is {s}"
                k.append(lookup.get(char, 0))  # filter invalid token
            ret.append(k)
        return ret


def decode(s):
    if isinstance(s, str):
        return sf.decoder(s)
    else:
        return [sf.decoder(i) for i in s]
