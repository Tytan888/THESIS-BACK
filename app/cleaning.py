import pandas as pd
import numpy as np
import re
import traceback

from difflib import SequenceMatcher
import unicodedata

from itertools import product
from fugashi import Tagger
from jamdict import Jamdict

from logger import ProcessingStep, log_gcp
from alignment import _calculate_labse_similarity 

fugashi_tagger = Tagger()
jmd = Jamdict()


class CleaningError(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code

# STEP 3: Unify the baseline and translation dataframes into one, even if the lines don't perfectly match up, using a modified Needleman-Wunsch algorithm that allows for matching one line in the baseline to multiple lines in the translation.

def _is_match_above_threshold(s1: str, s2: str, threshold: float) -> bool:
    s1 = s1.replace(" ", "").replace("\n", "")
    s2 = s2.replace(" ", "").replace("\n", "")
    similarity = SequenceMatcher(None, s1, s2).ratio() * 100
    return similarity >= threshold

def _calculate_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _normalize_text(text: str) -> str:
    return text.replace(" ", "").replace("\n", "").strip()

def _detect_edge_discrepancies(baseline: str, translation: str):
    baseline = _normalize_text(baseline)
    translation = _normalize_text(translation)

    s = SequenceMatcher(None, baseline, translation)
    match = s.find_longest_match(0, len(baseline), 0, len(translation))

    return {
        "missing_start": baseline[:match.a],
        "missing_end": baseline[match.a + match.size:],
        "extra_start": translation[:match.b],
        "extra_end": translation[match.b + match.size:]
    }

def _normalize_for_romaji_sim(text: str) -> str:
    result = []
    for c in text:
        code = ord(c)
        if 0x30A1 <= code <= 0x30F4:
            result.append(chr(code - 0x60))
        else:
            result.append(c)
    t = "".join(result)
    
    t = t.replace('（', '(').replace('）', ')').replace('！', '!').replace('？', '?')
    stripped = re.sub(r'[^\wぁ-んァ-ン一-龥a-zA-Z0-9]', '', t)
    
    if not stripped and t.strip():
        return t.strip().lower()
    return stripped.lower()

def _get_dp_alignment_indices(seqA: list, seqB: list, max_window: int = 5, gap_penalty: float = -0.2):
    n, m = len(seqA), len(seqB)
    score = np.zeros((n + 1, m + 1))
    traceback = {} 

    for i in range(1, n + 1): 
        score[i][0] = i * gap_penalty
        traceback[(i, 0)] = ("delete", 1)
    for j in range(1, m + 1): 
        score[0][j] = j * gap_penalty
        traceback[(0, j)] = ("insert", 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best_match_path_score = -np.inf
            best_w = 1
            
            for w in range(1, max_window + 1):
                if j - w < 0: continue
                candidate = "\n".join(seqB[j - w:j])
                s = _calculate_similarity(_normalize_text(seqA[i - 1]), _normalize_text(candidate))
                
                current_path_score = score[i - 1][j - w] + s
                
                if current_path_score > best_match_path_score:
                    best_match_path_score = current_path_score
                    best_w = w
            
            delete_score = score[i - 1][j] + gap_penalty
            insert_score = score[i][j - 1] + gap_penalty
            
            if best_match_path_score >= delete_score and best_match_path_score >= insert_score:
                score[i][j] = best_match_path_score
                traceback[(i, j)] = ("match", best_w)
            elif delete_score >= insert_score:
                score[i][j] = delete_score
                traceback[(i, j)] = ("delete", 1)
            else:
                score[i][j] = insert_score
                traceback[(i, j)] = ("insert", 1)

    i, j = n, m
    aligned_indices_A, aligned_indices_B = [], []

    while i > 0 or j > 0:
        action, w = traceback.get((i, j))
        
        if action == "match":
            aligned_indices_A.insert(0, i - 1)
            aligned_indices_B.insert(0, list(range(j - w, j)))
            i -= 1
            j -= w
        elif action == "delete":
            aligned_indices_A.insert(0, i - 1)
            aligned_indices_B.insert(0, None)
            i -= 1
        elif action == "insert":
            aligned_indices_A.insert(0, None)
            aligned_indices_B.insert(0, [j - 1])
            j -= 1

    return aligned_indices_A, aligned_indices_B

def unify_baseline_and_translation(baseline_df: pd.DataFrame, translation_df: pd.DataFrame, 
                                   col_baseline: str = "jp_baseline", col_translation: str = "jp_translation", 
                                   time_start_col: str = "start_time", time_end_col: str = "end_time", 
                                   max_window: int = 5, gap_penalty: float = -0.2) -> pd.DataFrame:
    
    seqA_base = baseline_df[col_baseline].tolist()
    seqB_trans = translation_df[col_translation].tolist()
    
    idxA_phase1, idxB_phase1 = _get_dp_alignment_indices(seqA_base, seqB_trans, max_window, gap_penalty)
    
    phase1_rows = []
    for idxA, idxB_list in zip(idxA_phase1, idxB_phase1):
        rowA = baseline_df.iloc[idxA] if idxA is not None else pd.Series({col: "" for col in baseline_df.columns})
        
        if idxB_list is not None:
            rowsB = translation_df.iloc[idxB_list]
            rowB = rowsB.apply(lambda col: " ".join(col.astype(str)), axis=0)
        else:
            rowB = pd.Series({col: "" for col in translation_df.columns})
            
        phase1_rows.append(pd.concat([rowA, rowB]))
        
    unified_phase1_df = pd.DataFrame(phase1_rows).reset_index(drop=True)

    seqA_trans = unified_phase1_df[col_translation].tolist()
    seqB_base = unified_phase1_df[col_baseline].tolist()
    
    idxA_phase2, idxB_phase2 = _get_dp_alignment_indices(seqA_trans, seqB_base, max_window, gap_penalty)
    
    final_rows = []
    for idxA, idxB_list in zip(idxA_phase2, idxB_phase2):
        if idxA is not None:
            base_row = unified_phase1_df.iloc[idxA].copy()
        else:
            base_row = pd.Series({col: "" for col in unified_phase1_df.columns})
            
        if idxB_list is not None:
            rowsB = unified_phase1_df.iloc[idxB_list]
            base_row[col_baseline] = " ".join(rowsB[col_baseline].astype(str))
            
            if time_start_col in rowsB.columns and time_end_col in rowsB.columns:
                base_row[time_start_col] = rowsB.iloc[0][time_start_col]
                base_row[time_end_col] = rowsB.iloc[-1][time_end_col]
        else:
            base_row[col_baseline] = ""
            
        final_rows.append(base_row)

    return pd.DataFrame(final_rows).reset_index(drop=True)

def merge_fragmented_rows(df: pd.DataFrame, baseline_col="jp_baseline", translation_col="jp_translation",
                          time_start_col="start_time", time_end_col="end_time", similarity_threshold=90.0) -> pd.DataFrame:
    final_df = df.copy()
    merged_rows = []
    skip_next = False
    n = len(final_df)

    for i in range(n):
        if skip_next:
            skip_next = False
            continue

        row1 = final_df.iloc[i].copy()
        if i < n - 1:
            row2 = final_df.iloc[i + 1]
            d1 = _detect_edge_discrepancies(row1[baseline_col].replace("　", ""), row1[translation_col].replace("　", ""))
            d2 = _detect_edge_discrepancies(row2[baseline_col].replace("　", ""), row2[translation_col].replace("　", ""))

            if d1["missing_end"] and d2["extra_start"]:
                if _is_match_above_threshold(d1["missing_end"], d2["extra_start"], similarity_threshold):
                    row1[baseline_col] = f"{row1[baseline_col]} {row2[baseline_col]}".strip()
                    row1[translation_col] = f"{row1[translation_col]} {row2[translation_col]}".strip()
                    row1[time_end_col] = row2[time_end_col]
                    for col in df.columns:
                        if col not in [baseline_col, translation_col, time_start_col, time_end_col]:
                            row1[col] = f"{row1[col]} {row2[col]}".strip()
                    skip_next = True

            elif d1["extra_end"] and d2["missing_start"]:
                if _is_match_above_threshold(d1["extra_end"], d2["missing_start"], similarity_threshold):
                    row1[baseline_col] = f"{row1[baseline_col]} {row2[baseline_col]}".strip()
                    row1[translation_col] = f"{row1[translation_col]} {row2[translation_col]}".strip()
                    row1[time_end_col] = row2[time_end_col]
                    for col in df.columns:
                        if col not in [baseline_col, translation_col, time_start_col, time_end_col]:
                            row1[col] = f"{row1[col]} {row2[col]}".strip()
                    skip_next = True

        merged_rows.append(row1)

    return pd.DataFrame(merged_rows).reset_index(drop=True)

def resolve_syntactic_inversions(df: pd.DataFrame, 
                                 baseline_col="jp_baseline", 
                                 translation_col="en_translation",
                                 time_start_col="start_time", 
                                 time_end_col="end_time", 
                                 inversion_threshold=0.20) -> pd.DataFrame:
    
    final_df = df.copy()
    resolved_rows = []
    skip_next = False
    n = len(final_df)

    for i in range(n):
        if skip_next:
            skip_next = False
            continue

        row1 = final_df.iloc[i].copy()
        
        if i < n - 1:
            row2 = final_df.iloc[i + 1]

            jp1, en1 = str(row1[baseline_col]), str(row1[translation_col])
            jp2, en2 = str(row2[baseline_col]), str(row2[translation_col])

            parallel_score_1 = _calculate_labse_similarity(jp1, en1)
            parallel_score_2 = _calculate_labse_similarity(jp2, en2)
            avg_parallel = (parallel_score_1 + parallel_score_2) / 2.0

            cross_score_1 = _calculate_labse_similarity(jp1, en2)
            cross_score_2 = _calculate_labse_similarity(jp2, en1)
            avg_cross = (cross_score_1 + cross_score_2) / 2.0

            if (avg_cross - avg_parallel) >= inversion_threshold:
                
                row1[baseline_col] = f"{jp1} {jp2}".strip()
                row1[translation_col] = f"{en1} {en2}".strip()
                row1[time_end_col] = row2[time_end_col]
                
                for col in df.columns:
                    if col not in [baseline_col, translation_col, time_start_col, time_end_col]:
                        row1[col] = f"{str(row1[col])} {str(row2[col])}".strip()
                        
                skip_next = True

        resolved_rows.append(row1)

    return pd.DataFrame(resolved_rows).reset_index(drop=True)

# STEP 4: Clean the unified dataframe by removing punctuation, normalizing spaces, and fixing common romaji exceptions.

def _pad_punctuation_with_spaces(s: str) -> str:
    always_space = ['"', '“', '”', '‘', '?', '.', ',', '—', '(', ')', "（", "）"] 
    for q in always_space:
        s = s.replace(q, f' {q} ')
    
    s = re.sub(r"(?<!\w)['’](?!\w)", r" \g<0> ", s)
    s = re.sub(r"(?<!\w),(?!\w)", r" \g<0> ", s)
    return ' '.join(s.split())

def _strip_punctuation(s: str) -> str:
    always_gone = ['"', '“', '”', '‘', '?', '.', ',', '\'', '’', '♡', '・'] 
    for q in always_gone:
        s = s.replace(q, ' ')
    return s

import re

def _strip_spaces(s: str) -> str:
    s = s.replace(' ', ' ')
    
    s = re.sub(r'(?<![a-zA-Z0-9])\s+(?![a-zA-Z0-9])', '', s)
    s = re.sub(r'\s+', ' ', s)
    
    return s.strip()

def _apply_romaji_exceptions(s: str) -> str:
    """Placeholder for manual overrides. Currently a no-op."""
    exceptions = [
        #("tama ni wa", "tamani wa"),
        #("hitori bocchi", "hitoribocchi"),
    ]
    for wrong, correct in exceptions:
        s = s.replace(wrong, correct)
    return s

def _find_infix_pattern(x: str, y: str):
    solutions = []
    for len_a in range(len(y)+1):
        a = y[:len_a]
        c = y[len_a:]
        b = x[len(a):len(x)-len(c)] 
        if a + b + c == x:
            solutions.append((a, b, c))
    return solutions

def normalize_lyric_text(unified_lyrics: pd.DataFrame) -> pd.DataFrame:
    cleaned = unified_lyrics.copy()

    cleaned["jp_baseline"] = cleaned["jp_baseline"].apply(_strip_spaces)
    cleaned["jp_baseline"] = cleaned["jp_baseline"].apply(_strip_punctuation)
    cleaned["jp_baseline"] = cleaned["jp_baseline"].apply(_pad_punctuation_with_spaces)

    cleaned["jp_translation"] = cleaned["jp_translation"].apply(_strip_spaces)
    cleaned["jp_translation"] = cleaned["jp_translation"].apply(_strip_punctuation)
    cleaned["jp_translation"] = cleaned["jp_translation"].apply(_pad_punctuation_with_spaces)

    cleaned["en_translation"] = cleaned["en_translation"].apply(_pad_punctuation_with_spaces)

    cleaned["ro_translation"] = cleaned["ro_translation"].apply(_strip_punctuation)
    cleaned["ro_translation"] = cleaned["ro_translation"].apply(lambda x: x.lower())
    cleaned["ro_translation"] = cleaned["ro_translation"].apply(_apply_romaji_exceptions)

    return cleaned

def resolve_infix_prefix_overlaps(df: pd.DataFrame, baseline_col="jp_baseline", translation_col="jp_translation") -> pd.DataFrame:
    final_df = df.copy()
    n = len(final_df)
    
    clean_pattern = re.compile(r'[^\wぁ-んァ-ン一-龥a-zA-Z0-9]')

    for i in range(n):
        baseline = clean_pattern.sub('', final_df.at[i, baseline_col])
        translation = clean_pattern.sub('', final_df.at[i, translation_col])

        if _find_infix_pattern(baseline, translation) or baseline.startswith(translation) or baseline.endswith(translation):
            final_df.at[i, baseline_col] = final_df.at[i, translation_col]

    return final_df.reset_index(drop=True)

# STEP 5: Align the cleaned baseline and translation dataframes using a modified Needleman-Wunsch algorithm that allows for matching one line in the baseline to multiple lines in the translation, but does not allow for dropping lines in the translation.

def _is_kanji(char: str) -> bool:
    try:
        return "CJK UNIFIED IDEOGRAPH" in unicodedata.name(char, "")
    except ValueError:
        return False

def _calculate_romaji_kanji_similarity(target_ro: str, cand_jp: str) -> float:
    t_ro = _normalize_for_romaji_sim(target_ro)
    c_jp = _normalize_for_romaji_sim(cand_jp)
    
    particle_overrides = {('わ', 'は'): 1.0, ('え', 'へ'): 1.0, ('お', 'を'): 1.0}
    if (t_ro, c_jp) in particle_overrides:
        return particle_overrides[(t_ro, c_jp)]
    
    if not t_ro or not c_jp: 
        return 0.0
    
    s = SequenceMatcher(None, t_ro, c_jp).ratio()
            
    return s

_romaji_to_hira_map = {
    "a":"あ","i":"い","u":"う","e":"え","o":"お",
    "ka":"か","ki":"き","ku":"く","ke":"け","ko":"こ",
    "kya":"きゃ","kyu":"きゅ","kyo":"きょ",
    "sa":"さ","shi":"し","su":"す","se":"せ","so":"そ",
    "sha":"しゃ","shu":"しゅ","sho":"しょ",
    "ta":"た","chi":"ち","tsu":"つ","te":"て","to":"と",
    "cha":"ちゃ","chu":"ちゅ","cho":"ちょ",
    "na":"な","ni":"に","nu":"ぬ","ne":"ね","no":"の",
    "nya":"にゃ","nyu":"にゅ","nyo":"にょ",
    "ha":"は","hi":"ひ","fu":"ふ","he":"へ","ho":"ほ",
    "hya":"ひゃ","hyu":"ひゅ","hyo":"ひょ",
    "ma":"ま","mi":"み","mu":"む","me":"め","mo":"も",
    "mya":"みゃ","myu":"みゅ","myo":"みょ",
    "ya":"や","yu":"ゆ","yo":"よ",
    "ra":"ら","ri":"り","ru":"る","re":"れ","ro":"ろ",
    "rya":"りゃ","ryu":"りゅ","ryo":"りょ",
    "wa":"わ","wo":"を",
    "ga":"が","gi":"ぎ","gu":"ぐ","ge":"げ","go":"ご",
    "gya":"ぎゃ","gyu":"ぎゅ","gyo":"ぎょ",
    "za":"ざ","ji":"じ","zu":"ず","ze":"ぜ","zo":"ぞ",
    "ja":"じゃ","ju":"じゅ","jo":"じょ",
    "da":"だ","di":"ぢ","du":"づ","de":"で","do":"ど",
    "dya":"ぢゃ","dyu":"ぢゅ","dyo":"ぢょ",
    "ba":"ば","bi":"び","bu":"ぶ","be":"べ","bo":"ぼ",
    "bya":"びゃ","byu":"びゅ","byo":"びょ",
    "pa":"ぱ","pi":"ぴ","pu":"ぷ","pe":"ぺ","po":"ぽ",
    "pya":"ぴゃ","pyu":"ぴゅ","pyo":"ぴょ",
    "n":"ん"
}

def _convert_romaji_to_hiragana(text: str) -> str:
    text = text.lower().replace("-", "")
    result = ""
    i = 0
    while i < len(text):
        if text[i:i+2] == "nn":
            result += "ん"
            i += 1
            continue

        matched = False
        for l in (3,2,1):
            if i+l <= len(text) and text[i:i+l] in _romaji_to_hira_map:
                result += _romaji_to_hira_map[text[i:i+l]]
                i += l
                matched = True
                break
        if matched:
            continue

        if i+1 < len(text) and text[i] == text[i+1] and text[i] not in "aeioun":
            result += "っ"
            i += 1
            continue

        if text[i] == "n":
            if i+1 == len(text) or text[i+1] not in "aiueoyn":
                result += "ん"
                i += 1
                continue

        result += text[i]
        i += 1
    return result

def _kata_to_hira(text: str) -> str:
    result = []
    for c in text:
        code = ord(c)
        if 0x30A1 <= code <= 0x30F4:
            result.append(chr(code - 0x60))
        else:
            result.append(c)
    return "".join(result)

def align_kanji_and_romaji_tokens(df: pd.DataFrame, max_window=6, gap_penalty=-0.8, debug_idx=-1) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["jp_final"] = ""
    df_copy["ro_final"] = ""

    for idx, row in df_copy.iterrows():
        jp_text = row["jp_baseline"]
        ro_text = row["ro_translation"]

        if not jp_text or not ro_text:
            continue

        raw_tokens = []
        for tok in fugashi_tagger(jp_text):
            surface = tok.surface.strip()
            if not surface or surface == "<EOS>": continue
            
            pronunciation = getattr(tok.feature, 'kana', None)
            if not pronunciation:
                pronunciation = surface 
            
            if surface in ['ー', '・', '~', '〜', '-'] and raw_tokens:
                raw_tokens[-1]["surface"] += surface
                raw_tokens[-1]["fuga_pron"] += surface
            else:
                raw_tokens.append({"surface": surface, "fuga_pron": pronunciation})

        jp_tokens_info = []
        for tok_data in raw_tokens:
            surface = tok_data["surface"]
            has_k = any(_is_kanji(c) for c in surface)
            
            fuga_pron_hira = _kata_to_hira(tok_data["fuga_pron"])
            
            readings = set([_convert_romaji_to_hiragana(fuga_pron_hira)])

            if has_k:
                result = jmd.lookup(surface)
                for entry in result.entries:
                    if any(k.text == surface for k in entry.kanji_forms):
                        for kana in entry.kana_forms:
                            normalized_kana = _kata_to_hira(kana.text)
                            readings.add(normalized_kana)

            particle_exceptions = {'は': 'わ', 'へ': 'え', 'を': 'お'}
            if surface in particle_exceptions:
                readings.add(particle_exceptions[surface])

            jp_tokens_info.append({
                "surface": surface,
                "readings": list(readings)
            })

        if debug_idx >= 0 and idx == debug_idx:
            print(jp_tokens_info)

        spaced_ro = ro_text
        for punc in ['(', ')', '（', '）', '!', '?', '！', '？', ',', '.', '、', '。']:
            spaced_ro = spaced_ro.replace(punc, f' {punc} ')
            
        ro_originals = [r for r in spaced_ro.split() if r.strip() and r != "<EOS>"]
        ro_hiras = [_convert_romaji_to_hiragana(r) for r in ro_originals]

        n, m = len(ro_hiras), len(jp_tokens_info)
        score = np.full((n + 1, m + 1), -1000.0) 
        backtrack = {}

        score[0, 0] = 0
        for i in range(1, n + 1): 
            score[i, 0] = i * gap_penalty
            backtrack[(i, 0)] = (i - 1, 0, 1, 0) 
            
        for j in range(1, m + 1): 
            score[0, j] = j * gap_penalty
            backtrack[(0, j)] = (0, j - 1, 0, 1)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                for r_win in range(1, min(i, max_window) + 1):
                    for j_win in range(1, min(j, max_window) + 1):
                        
                        target_ro = "".join(ro_hiras[i-r_win:i])
                        jp_slice = jp_tokens_info[j-j_win:j]
                        
                        reading_lists = [t['readings'] for t in jp_slice]
                        candidates = [''.join(p) for p in product(*reading_lists)]
                        
                        s = max(_calculate_romaji_kanji_similarity(target_ro, cand) for cand in candidates)
                        
                        base_score = s * (r_win + j_win)
                        window_penalty = 0.5 * (r_win * j_win - 1)
                        match_score = base_score - window_penalty
                        
                        current_val = score[i-r_win, j-j_win] + match_score
                        
                        if current_val > score[i, j]:
                            score[i, j] = current_val
                            backtrack[(i, j)] = (i-r_win, j-j_win, r_win, j_win)

                if score[i-1, j] + gap_penalty > score[i, j]:
                    score[i, j] = score[i-1, j] + gap_penalty
                    backtrack[(i, j)] = (i-1, j, 1, 0)
                if score[i, j-1] + gap_penalty > score[i, j]:
                    score[i, j] = score[i, j-1] + gap_penalty
                    backtrack[(i, j)] = (i, j-1, 0, 1)

        i, j = n, m
        res_ro, res_jp = [], []
        while i > 0 or j > 0:
            if (i, j) in backtrack:
                pi, pj, rw, jw = backtrack[(i, j)]
                
                ro_segment = "".join(ro_originals[i-rw:i]) if rw > 0 else ""
                
                if jw > 0:
                    jp_words = [t['surface'] for t in jp_tokens_info[j-jw:j]]
                    jp_segment = jp_words[ 0 ]
                    for w_idx in range(1, len(jp_words)):
                        prev = jp_words[w_idx - 1]
                        curr = jp_words[w_idx]
                        
                        if prev and curr and prev[-1].isascii() and prev[-1].isalnum() and curr[ 0 ].isascii() and curr[ 0 ].isalnum():
                            jp_segment += " "
                        jp_segment += curr
                else:
                    jp_segment = ""
                
                if ro_segment or jp_segment:
                    res_ro.insert(0, ro_segment)
                    res_jp.insert(0, jp_segment)
                i, j = pi, pj
            else:
                break

        df_copy.at[idx, "jp_final"] = " ".join(filter(None, res_jp))
        df_copy.at[idx, "ro_final"] = " ".join(filter(None, res_ro))

    return df_copy

def run_cleaning_pipeline(baseline_df: pd.DataFrame, translation_df: pd.DataFrame, progress_cb=None) -> pd.DataFrame:
    try:
        # STEP 3: Initial Row Alignments
        if progress_cb: progress_cb(ProcessingStep.UNIFYING_LYRICS)
        unified = unify_baseline_and_translation(baseline_df, translation_df)
        unified = merge_fragmented_rows(unified)
        unified = resolve_syntactic_inversions(unified)
    except Exception as e:
        log_gcp("ERROR", f"Error during initial alignment and merging... {e}", traceback=traceback.format_exc())
        raise CleaningError("Error during initial alignment and merging...", "ERR_MERGE_FAILED")

    try:
        # STEP 4: Cleaning & Overlaps
        cleaned = normalize_lyric_text(unified)
        cleaned = resolve_infix_prefix_overlaps(cleaned)
    except Exception as e:
        log_gcp("ERROR", f"Error during cleaning and removal of overlaps... {e}", traceback=traceback.format_exc())
        raise CleaningError("Error during cleaning and removal of overlaps...", "ERR_CLEANING_FAILED")
    try:
        # STEP 5: Token Matching
        if progress_cb: progress_cb(ProcessingStep.CLEANING_TEXT)
        final_cleaned_df = align_kanji_and_romaji_tokens(cleaned)
        
        valid_starts = final_cleaned_df["start_time"].fillna("").astype(str).str.strip() != ""
        valid_ends = final_cleaned_df["end_time"].fillna("").astype(str).str.strip() != ""
        
        final_cleaned_df = final_cleaned_df[valid_starts & valid_ends].copy()
        final_cleaned_df = final_cleaned_df.reset_index(drop=True)

    except Exception as e:
        log_gcp("ERROR", f"Error during token matching... {e}", traceback=traceback.format_exc())
        raise CleaningError("Error during token matching...", "ERR_CLEANING_FAILED")

    return final_cleaned_df