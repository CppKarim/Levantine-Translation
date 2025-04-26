#!/usr/bin/env python3
"""
Arabic Transliteration Library
A Python port of the JavaScript Arabic transliteration system
Adapted into python from this repo:
https://github.com/levantinedictionary/levantinetransliterator/tree/main
"""
import re
from typing import List, Dict, Union, Tuple, Set


class TreeNode:
    """Tree node for tracking possible transliteration paths"""
    
    def __init__(self, value: str, parent=None):
        self.value = value
        self.children = []
        self.parent = parent
    
    def add_child(self, value: str):
        """Add a single child node"""
        self.children.append(TreeNode(self.value + value, self))
    
    def add_children(self, values: List[str]):
        """Add multiple children from a list of values"""
        for value in values:
            self.children.append(TreeNode(self.value + value, self))
    
    def add_siblings(self, values: List[str]):
        """Add siblings through the parent node"""
        if self.parent:
            self.parent.add_children(values)
    
    def add_sibling(self, value: str):
        """Add a single sibling through the parent node"""
        if self.parent:
            self.parent.add_child(value)


# Arabic character constants
ALEF = "ا"
BAA2 = "ب"
PAA2 = "پ"
VAA2 = "ڤ"
TAA2 = "ت"
GAAL = "چ"
TA2_MARBOUTA = "ة"
THAA = "ط"
ALEF_HAMZE = "أ"
QAF = "ق"
HAMZE_SEAT = "إ"
KAAF = "ك"
SHADDE = "\u0651"
H7A2 = "ح"
GHAYN = "غ"
AAYN = "ع"
YAA2 = "ي"
FAT7A = "\u064E"
KHA2 = "خ"
SEEN = "س"
SHEEN = "ش"
DAAL = "د"
DHAAD = "ض"
SAAD = "ص"
THA = "ط"
NOON = "ن"
FAA2 = "ف"
JIIM = "ج"
HAA2 = "ه"
LAAM = "ل"
MIIM = "م"
WAAW = "و"
DAMME = "\u064F"
RAA2 = "ر"
ZHAA2 = "ظ"
ZHAAL = "ذ"
ZAY = "ز"
ALEF_MAKSOURA = "ى"
HAMZE_SATER = "ء"
KASRA = "\u0650"
INITIAL = "ـ"
TANWEEN_FAT7A = "اً"


# Transliteration rule sets
ABZ_TO_ARB = [
    # S = start, M = 'middle', E = 'end', A = 'any position'
    {"rule": "S", "l": "aa", "a": [ALEF]},
    {"rule": "M", "l": "aa", "a": [ALEF]},
    {"rule": "S", "l": "a", "a": [ALEF_HAMZE, QAF]},
    {"rule": "S", "l": "A", "a": [ALEF_HAMZE]},
    {"rule": "M", "l": "a", "a": [FAT7A, ALEF]},
    {"rule": "S", "l": "2", "a": [ALEF]},
    {"rule": "M", "l": "2", "a": [QAF, ALEF_HAMZE]},
    {"rule": "E", "l": "2", "a": [QAF]},
    {"rule": "M", "l": "22", "a": [f"{QAF}{SHADDE}"]},
    {"rule": "S", "l": "2", "a": [ALEF]},
    {"rule": "M", "l": "2", "a": [ALEF]},
    {"rule": "E", "l": "an", "a": [f"{TANWEEN_FAT7A}", f"{FAT7A}{NOON}"]},
    {"rule": "A", "l": "7", "a": [H7A2]},
    {"rule": "A", "l": "*", "a": [SHADDE]},
    {"rule": "A", "l": "77", "a": [f"{H7A2}{SHADDE}"]},
    {"rule": "A", "l": "3", "a": [AAYN]},
    {"rule": "E", "l": " 3", "a": [f" {AAYN}{INITIAL}"]},
    {"rule": "E", "l": " 3a", "a": [f" {AAYN}{FAT7A}{INITIAL}"]},
    {"rule": "E", "l": " t", "a": [f" {TAA2}{INITIAL}"]},
    {"rule": "E", "l": " ta", "a": [f" {TAA2}{FAT7A}{INITIAL}"]},
    {"rule": "E", "l": " b", "a": [f" {BAA2}{INITIAL}"]},
    {"rule": "E", "l": " bi", "a": [f" {BAA2}{KASRA}{INITIAL}"]},
    {"rule": "E", "l": " l", "a": [f" {LAAM}{INITIAL}"]},
    {"rule": "E", "l": " la", "a": [f" {LAAM}{FAT7A}{INITIAL}"]},
    {"rule": "E", "l": " 7", "a": [f" {H7A2}{INITIAL}"]},
    {"rule": "E", "l": " 7a", "a": [f" {H7A2}{FAT7A}{INITIAL}"]},
    {"rule": "A", "l": "33", "a": [f"{GHAYN}{SHADDE}"]},
    {"rule": "A", "l": " ", "a": [" "]},
    {"rule": "A", "l": "kh", "a": [KHA2]},
    {"rule": "A", "l": "5", "a": [KHA2]},
    {"rule": "A", "l": "55", "a": [f"{KHA2}{SHADDE}"]},
    {"rule": "A", "l": "aa", "a": [ALEF]},
    {"rule": "A", "l": "b", "a": [BAA2]},
    {"rule": "A", "l": "bb", "a": [f"{BAA2}{SHADDE}"]},
    {"rule": "A", "l": "c", "a": [SEEN]},
    {"rule": "A", "l": "d", "a": [DAAL, DHAAD]},
    {"rule": "A", "l": "D", "a": [DHAAD]},
    {"rule": "A", "l": "dd", "a": [f"{DAAL}{SHADDE}", f"{DHAAD}{SHADDE}"]},
    {"rule": "E", "l": "e", "a": [f"{KASRA}{TA2_MARBOUTA}", YAA2]},
    {"rule": "M", "l": "e", "a": [KASRA, f"{KASRA}{ALEF}"]},
    {"rule": "A", "l": "ee", "a": [YAA2, f"{ALEF}{KASRA}"]},
    {"rule": "S", "l": "e", "a": [HAMZE_SEAT, QAF]},
    {"rule": "A", "l": " e", "a": [f" {ALEF}"]},
    {"rule": "A", "l": "f", "a": [FAA2]},
    {"rule": "A", "l": "g", "a": [JIIM, GAAL]},
    {"rule": "A", "l": "gg", "a": [f"{JIIM}{SHADDE}", f"{GAAL}{SHADDE}"]},
    {"rule": "A", "l": "gh", "a": [GHAYN]},
    {"rule": "A", "l": "h", "a": [HAA2]},
    {"rule": "A", "l": "ii", "a": [YAA2]},
    {"rule": "M", "l": "i", "a": [KASRA, YAA2]},
    {"rule": "E", "l": "i", "a": [YAA2]},
    {"rule": "S", "l": "i", "a": [HAMZE_SEAT]},
    {"rule": "A", "l": "j", "a": [JIIM]},
    {"rule": "A", "l": "k", "a": [KAAF, QAF]},
    {"rule": "A", "l": "kk", "a": [f"{KAAF}{SHADDE}"]},
    {"rule": "A", "l": "l", "a": [LAAM]},
    {"rule": "A", "l": "ll", "a": [f"{LAAM}{SHADDE}"]},
    {"rule": "A", "l": "m", "a": [MIIM]},
    {"rule": "A", "l": "M", "a": [MIIM]},
    {"rule": "A", "l": "mm", "a": [f"{MIIM}{SHADDE}"]},
    {"rule": "A", "l": "mmu", "a": [f"{MIIM}{MIIM}{DAMME}"]},
    {"rule": "A", "l": "n", "a": [NOON]},
    {"rule": "A", "l": "nn", "a": [f"{NOON}{SHADDE}"]},
    {"rule": "M", "l": "o", "a": [DAMME, WAAW]},
    {"rule": "E", "l": "o", "a": [DAMME, WAAW]},
    {"rule": "M", "l": "o", "a": [WAAW]},
    {"rule": "E", "l": "o", "a": [WAAW]},
    {"rule": "S", "l": "o", "a": ["اُ"]},
    {"rule": "A", "l": "p", "a": [PAA2]},
    {"rule": "A", "l": "pp", "a": [f"{PAA2}{SHADDE}"]},
    {"rule": "A", "l": "q", "a": [QAF]},
    {"rule": "A", "l": "r", "a": [RAA2]},
    {"rule": "A", "l": "rr", "a": [f"{RAA2}{SHADDE}"]},
    {"rule": "A", "l": "s", "a": [SEEN, SAAD]},
    {"rule": "A", "l": "S", "a": [SAAD]},
    {"rule": "A", "l": "ss", "a": [f"{SEEN}{SHADDE}", f"{SAAD}{SHADDE}"]},
    {"rule": "A", "l": "t", "a": [TAA2, THAA]},
    {"rule": "A", "l": "T", "a": [THAA]},
    {"rule": "A", "l": "TT", "a": [f"{THAA}{SHADDE}"]},
    {"rule": "A", "l": "tt", "a": [f"{TAA2}{SHADDE}", THAA]},
    {"rule": "A", "l": "v", "a": [VAA2]},
    {"rule": "A", "l": "vv", "a": [f"{VAA2}{SHADDE}"]},
    {"rule": "A", "l": "u", "a": [WAAW, DAMME]},
    {"rule": "A", "l": "uu", "a": [f"{DAMME}{WAAW}"]},
    {"rule": "A", "l": "ou", "a": [WAAW]},
    {"rule": "A", "l": "oo", "a": [WAAW]},
    {"rule": "A", "l": "w", "a": [WAAW]},
    {"rule": "E", "l": "u", "a": [WAAW]},
    {"rule": "A", "l": "ww", "a": [f"{WAAW}{SHADDE}"]},
    {"rule": "A", "l": "x", "a": [f"{KAAF}{SEEN}"]},
    {"rule": "A", "l": "y", "a": [YAA2]},
    {"rule": "A", "l": "yy", "a": [f"{YAA2}{SHADDE}"]},
    {"rule": "A", "l": "z", "a": [ZAY, ZHAA2, ZHAAL]},
    {"rule": "A", "l": "Z", "a": [ZHAA2]},
    {"rule": "A", "l": "sh", "a": [SHEEN]},
    {"rule": "A", "l": f"{SHEEN}{SHEEN}", "a": [f"{SHEEN}{SHADDE}"]},
    {"rule": "A", "l": "sh", "a": [f"{SEEN}{HAA2}"]},
    {"rule": "A", "l": "ch", "a": [SHEEN]},
    {"rule": "E", "l": "aa", "a": [ALEF_MAKSOURA]},
    {"rule": "E", "l": "a", "a": [TA2_MARBOUTA, ALEF_MAKSOURA, ALEF]},
]

TO_PRON_RULES = [
    # Phonetic transliteration rules
    {"rule": "A", "l": "2", "a": ["ʔ"]},
    {"rule": "A", "l": "b", "a": ["b"]},
    {"rule": "A", "l": "t", "a": ["t", "ṭ"]},
    {"rule": "A", "l": "th", "a": ["ṯ", "th"]},
    {"rule": "A", "l": "TH", "a": ["ṯ"]},
    {"rule": "A", "l": "ث", "a": ["ṯ"]},
    {"rule": "A", "l": "j", "a": ["ǧ"]},
    {"rule": "A", "l": "g", "a": ["ǧ", "g"]},
    {"rule": "A", "l": JIIM, "a": ["ǧ"]},
    {"rule": "A", "l": "7", "a": ["ḥ"]},
    {"rule": "A", "l": H7A2, "a": ["ḥ"]},
    {"rule": "A", "l": "kh", "a": ["ẖ", "kh"]},
    {"rule": "A", "l": "KH", "a": ["ẖ"]},
    {"rule": "A", "l": "5", "a": ["ẖ"]},
    {"rule": "A", "l": KHA2, "a": ["ẖ"]},
    {"rule": "A", "l": "d", "a": ["d"]},
    {"rule": "A", "l": "z", "a": ["ḏ"]},
    {"rule": "A", "l": ZHAAL, "a": ["ḏ"]},
    {"rule": "A", "l": "r", "a": ["r"]},
    {"rule": "A", "l": "z", "a": ["z"]},
    {"rule": "A", "l": "s", "a": ["s"]},
    {"rule": "A", "l": "c", "a": ["s"]},
    {"rule": "A", "l": "s", "a": ["ṣ"]},
    {"rule": "A", "l": "S", "a": ["ṣ"]},
    {"rule": "A", "l": SAAD, "a": ["ṣ"]},
    {"rule": "A", "l": "sh", "a": ["š", "sh"]},
    {"rule": "A", "l": "SH", "a": ["š"]},
    {"rule": "A", "l": SHEEN, "a": ["š"]},
    {"rule": "A", "l": "d", "a": ["ḍ"]},
    {"rule": "A", "l": "D", "a": ["ḏ"]},
    {"rule": "A", "l": "T", "a": ["ṭ"]},
    {"rule": "A", "l": THAA, "a": ["ṭ"]},
    {"rule": "A", "l": "z", "a": ["ẓ"]},
    {"rule": "A", "l": "Z", "a": ["ẓ"]},
    {"rule": "A", "l": "3", "a": ["ʕ"]},
    {"rule": "A", "l": "ʿ", "a": ["ʕ"]},
    {"rule": "A", "l": "gh", "a": ["ġ"]},
    {"rule": "A", "l": "f", "a": ["f"]},
    {"rule": "A", "l": "q", "a": ["q"]},
    {"rule": "A", "l": "k", "a": ["k"]},
    {"rule": "A", "l": "l", "a": ["l"]},
    {"rule": "A", "l": "m", "a": ["m"]},
    {"rule": "A", "l": "n", "a": ["n"]},
    {"rule": "A", "l": "h", "a": ["h"]},
    {"rule": "A", "l": "w", "a": ["w"]},
    {"rule": "A", "l": "y", "a": ["y"]},
    {"rule": "A", "l": "i", "a": ["i"]},
    {"rule": "A", "l": "ii", "a": ["ī"]},
    {"rule": "A", "l": "I", "a": ["ī"]},
    {"rule": "A", "l": "aa", "a": ["ā"]},
    {"rule": "A", "l": "ee", "a": ["ē"]},
    {"rule": "A", "l": "é", "a": ["ē"]},
    {"rule": "A", "l": "oo", "a": ["ō"]},
    {"rule": "A", "l": "e", "a": ["ə", "e"]},
    {"rule": "A", "l": "uu", "a": ["ū"]},
    {"rule": "A", "l": "y", "a": ["y"]},
]

PRON_TO_ARB_RULES = [
    # Phonetic to Arabic script rules
    {"rule": "A", "l": "b", "a": [BAA2]},
    {"rule": "A", "l": "p", "a": [PAA2]},
    {"rule": "A", "l": "bb", "a": [f"{BAA2}{SHADDE}"]},
    {"rule": "A", "l": "t", "a": [TAA2]},
    {"rule": "A", "l": "tt", "a": ["تّ"]},
    {"rule": "A", "l": "ṯ", "a": ["ث"]},
    {"rule": "A", "l": "ṯṯ", "a": ["ثّ"]},
    {"rule": "A", "l": "g", "a": [GAAL]},
    {"rule": "A", "l": "gg", "a": [f"{GAAL}{SHADDE}"]},
    {"rule": "A", "l": "v", "a": [VAA2]},
    {"rule": "A", "l": "ǧ", "a": [JIIM]},
    {"rule": "A", "l": "ǧǧ", "a": ["جّ"]},
    {"rule": "A", "l": "ḥ", "a": [H7A2]},
    {"rule": "A", "l": "ẖ", "a": [KHA2]},
    {"rule": "A", "l": "ẖẖ", "a": ["خّ"]},
    {"rule": "A", "l": "d", "a": [DAAL]},
    {"rule": "A", "l": "dd", "a": ["دّ"]},
    {"rule": "A", "l": "ḏ", "a": [ZHAAL]},
    {"rule": "A", "l": "ḏḏ", "a": ["ذّ"]},
    {"rule": "A", "l": "r", "a": [RAA2]},
    {"rule": "A", "l": "rr", "a": ["رّ"]},
    {"rule": "A", "l": "z", "a": [ZAY]},
    {"rule": "A", "l": "zz", "a": ["زّ"]},
    {"rule": "A", "l": "s", "a": [SEEN]},
    {"rule": "A", "l": "ss", "a": ["سّ"]},
    {"rule": "A", "l": "š", "a": [SHEEN]},
    {"rule": "A", "l": "šš", "a": ["شّ"]},
    {"rule": "A", "l": "ṣ", "a": [SAAD]},
    {"rule": "A", "l": "ṣṣ", "a": ["صّ"]},
    {"rule": "A", "l": "ḍ", "a": [DHAAD]},
    {"rule": "A", "l": "ḍḍ", "a": ["ضّ"]},
    {"rule": "A", "l": "ṭ", "a": [THAA]},
    {"rule": "A", "l": "ṭṭ", "a": ["طّ"]},
    {"rule": "A", "l": "ẓ", "a": ["ظ"]},
    {"rule": "A", "l": "ẓẓ", "a": ["ظّ"]},
    {"rule": "A", "l": "ʕ", "a": [AAYN]},
    {"rule": "A", "l": "ʕʕ", "a": ["عّ"]},
    {"rule": "A", "l": "ġ", "a": [GHAYN]},
    {"rule": "A", "l": "ġġ", "a": ["غّ"]},
    {"rule": "A", "l": "f", "a": [FAA2]},
    {"rule": "A", "l": "ff", "a": ["فّ"]},
    {"rule": "A", "l": "q", "a": [QAF]},
    {"rule": "A", "l": "qq", "a": ["قّ"]},
    {"rule": "A", "l": "k", "a": [KAAF]},
    {"rule": "A", "l": "kk", "a": [f"{KAAF}{SHADDE}"]},
    {"rule": "A", "l": "l", "a": [LAAM]},
    {"rule": "A", "l": "ll", "a": [f"{LAAM}{SHADDE}"]},
    {"rule": "A", "l": "m", "a": [MIIM]},
    {"rule": "A", "l": "mm", "a": [f"{MIIM}{SHADDE}"]},
    {"rule": "A", "l": "n", "a": [NOON]},
    {"rule": "A", "l": "nn", "a": [f"{NOON}{SHADDE}"]},
    {"rule": "A", "l": "h", "a": [HAA2]},
    {"rule": "A", "l": "hh", "a": [f"{HAA2}{SHADDE}"]},
    {"rule": "A", "l": "w", "a": [WAAW]},
    {"rule": "A", "l": "ww", "a": [f"{WAAW}{SHADDE}"]},
    {"rule": "A", "l": "y", "a": [YAA2]},
    {"rule": "A", "l": "yy", "a": [f"{YAA2}{SHADDE}"]},
    {"rule": "S", "l": "a", "a": [f"{ALEF}{FAT7A}"]},
    {"rule": "M", "l": "a", "a": [FAT7A]},
    {"rule": "E", "l": "a", "a": [FAT7A]},
    {"rule": "S", "l": "e", "a": [f"{ALEF}{KASRA}"]},
    {"rule": "M", "l": "e", "a": [KASRA]},
    {"rule": "E", "l": "e", "a": [KASRA]},
    {"rule": "A", "l": "i", "a": [KASRA]},
    {"rule": "S", "l": "ə", "a": [f"{ALEF}{KASRA}"]},
    {"rule": "M", "l": "ə", "a": [KASRA]},
    {"rule": "E", "l": "ə", "a": [KASRA]},
    {"rule": "A", "l": "o", "a": [DAMME]},
    {"rule": "A", "l": "u", "a": [DAMME]},
    {"rule": "A", "l": "ē", "a": [f"{KASRA}{ALEF}"]},
    {"rule": "A", "l": "ā", "a": [ALEF]},
    {"rule": "A", "l": "ō", "a": [WAAW]},
    {"rule": "A", "l": "ū", "a": [WAAW]},
    {"rule": "A", "l": "ī", "a": [YAA2]},
    {"rule": "A", "l": "ʔ", "a": [HAMZE_SATER]},
]


def l_to_a(prev: str, prev_prev: str, current: str, c_type: str, rules: List[Dict]) -> Dict:
    """
    Find possible transliterations for a character based on context.
    
    Args:
        prev: Previous character
        prev_prev: Character before previous
        current: Current character
        c_type: Position type (S=start, M=middle, E=end, A=any)
        rules: List of transliteration rules
        
    Returns:
        Dictionary with possible children, siblings, and step-siblings
    """
    filtered_rules = [r for r in rules if r["rule"] == c_type or r["rule"] == "A"]
    
    # Find direct matches for current character
    children = []
    for rule in filtered_rules:
        if rule["l"] == current:
            children.extend(rule["a"])
    
    if not children:
        children = [current]
    
    # Find matches for prev+current
    siblings = []
    if prev:
        pattern = f"{prev}{current}"
        for rule in filtered_rules:
            if rule["l"] == pattern:
                siblings.extend(rule["a"])
    
    # Find matches for prev_prev+prev+current
    step_siblings = []
    if prev and prev_prev:
        pattern = f"{prev_prev}{prev}{current}"
        for rule in filtered_rules:
            if rule["l"] == pattern:
                step_siblings.extend(rule["a"])
    
    return {
        "children": children,
        "siblings": siblings,
        "step_siblings": step_siblings
    }


def uniq_by(arr: List, key_func=None) -> List:
    """
    Remove duplicates from a list based on a key function.
    
    Args:
        arr: List to process
        key_func: Function to generate keys for comparison
        
    Returns:
        List with duplicates removed
    """
    if not key_func:
        key_func = lambda x: x
        
    seen = set()
    result = []
    
    for item in arr:
        key = item if (item is None or item is True or item is False) else key_func(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
            
    return result


def to_tree(word_src: str, rules: List[Dict], limit: int = 100) -> List[str]:
    """
    Build a tree of possible transliterations and return leaf values.
    
    Args:
        word_src: Source word to transliterate
        rules: List of transliteration rules
        limit: Maximum number of candidates to consider
        
    Returns:
        List of possible transliterations
    """
    # Replace three or more consecutive identical characters with just two
    word = re.sub(r'(.)\1{2,}', r'\1\1', word_src)
    
    root = TreeNode("", None)
    leaves = [root]
    end_index = len(word) - 1
    
    prev = None
    prev_prev = None
    
    for i, current in enumerate(word):
        # Determine position type
        if i == 0:
            pos_type = "S"  # Start
        elif i == end_index:
            pos_type = "E"  # End
        else:
            pos_type = "M"  # Middle
        
        # Get possible transliterations
        sugg = l_to_a(prev, prev_prev, current, pos_type, rules)
        
        # Update tree based on suggestions
        if sugg["step_siblings"]:
            # Handle three-character patterns
            parents = [leaf.parent for leaf in leaves]
            for p in parents:
                p.children = []
            for p in parents:
                p.add_children(sugg["step_siblings"])
            leaves = []
            for p in parents:
                leaves.extend(p.children)
        elif sugg["siblings"]:
            # Handle two-character patterns
            parents = [leaf.parent for leaf in leaves]
            for p in parents:
                p.children = []
            for p in parents:
                p.add_children(sugg["siblings"])
            leaves = []
            for p in parents:
                leaves.extend(p.children)
        else:
            # Handle single character patterns
            for leaf in leaves:
                leaf.add_children(sugg["children"])
            temp = []
            for leaf in leaves:
                temp.extend(leaf.children)
            leaves = temp
        
        # Update context for next iteration
        prev_prev = prev
        prev = current
        
        # Reduce duplication and control tree size
        leaves = uniq_by(leaves, lambda e: e.value)
        
        if len(leaves) > limit:
            # If we have too many possibilities, sample every other one
            leaves = [leaf for i, leaf in enumerate(leaves) if i % 2 == 0]
    
    # Extract values from leaf nodes
    results = [leaf.value for leaf in leaves if leaf.value]
    
    return results


class ArabicTransliterator:
    """Main class for Arabic transliteration operations"""
    
    def __init__(self):
        """Initialize the transliterator with default rule sets"""
        pass
        
    def transliterate(self, word: str, limit: int = 100) -> List[str]:
        """
        Convert from Latin script to Arabic script.
        
        Args:
            word: Latin script text to convert
            limit: Maximum number of transliteration candidates
            
        Returns:
            List of possible Arabic script transliterations
        """
        return to_tree(word, ABZ_TO_ARB, limit)
    
    def pronunciate(self, word: str, limit: int = 100) -> List[str]:
        """
        Convert from Latin script to phonetic notation.
        
        Args:
            word: Latin script text to convert
            limit: Maximum number of transliteration candidates
            
        Returns:
            List of possible phonetic representations
        """
        return to_tree(word, TO_PRON_RULES, limit)
    
    def to_arb(self, word: str, limit: int = 2) -> List[str]:
        """
        Convert from phonetic notation to Arabic script.
        
        Args:
            word: Phonetic notation to convert
            limit: Maximum number of transliteration candidates
            
        Returns:
            List of possible Arabic script transliterations
        """
        return to_tree(word, PRON_TO_ARB_RULES, limit)
    
    def custom(self, word: str, rules: List[Dict], limit: int = 100) -> List[str]:
        """
        Apply custom transliteration rules.
        
        Args:
            word: Text to convert
            rules: Custom rule set to apply
            limit: Maximum number of transliteration candidates
            
        Returns:
            List of possible transliterations
        """
        return to_tree(word, rules, limit)