import subprocess
import os
import traceback
import tempfile
import random

import spacy
from fugashi import Tagger
from jamdict import Jamdict
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from openai import OpenAI, APITimeoutError
from google import genai
from google.genai import types

from logger import ProcessingStep, log_gcp

tagger = Tagger()
nlp = spacy.load("en_core_web_sm")
jam = Jamdict()

from logger import log_gcp


class AlignmentError(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code

HIRA_TO_ROMAJI = {
    'きゃ':'kya', 'きゅ':'kyu', 'きょ':'kyo', 'しゃ':'sha', 'しゅ':'shu', 'しょ':'sho',
    'ちゃ':'cha', 'ちゅ':'chu', 'ちょ':'cho', 'にゃ':'nya', 'にゅ':'nyu', 'にょ':'nyo',
    'ひゃ':'hya', 'ひゅ':'hyu', 'ひょ':'hyo', 'みゃ':'mya', 'みゅ':'myu', 'みょ':'myo',
    'りゃ':'rya', 'りゅ':'ryu', 'りょ':'ryo', 'ぎゃ':'gya', 'ぎゅ':'gyu', 'ぎょ':'gyo',
    'じゃ':'ja', 'じゅ':'ju', 'じょ':'jo', 'びゃ':'bya', 'びゅ':'byu', 'びょ':'byo',
    'ぴゃ':'pya', 'ぴゅ':'pyu', 'ぴょ':'pyo', 'あ':'a', 'い':'i', 'う':'u', 'え':'e', 'お':'o',
    'か':'ka', 'き':'ki', 'く':'ku', 'け':'ke', 'こ':'ko', 'さ':'sa', 'し':'shi', 'す':'su', 'せ':'se', 'そ':'so',
    'た':'ta', 'ち':'chi', 'つ':'tsu', 'て':'te', 'と':'to', 'な':'na', 'に':'ni', 'ぬ':'nu', 'ね':'ne', 'の':'no',
    'は':'ha', 'ひ':'hi', 'ふ':'fu', 'へ':'he', 'ほ':'ho', 'ま':'ma', 'み':'mi', 'む':'mu', 'め':'me', 'も':'mo',
    'や':'ya', 'ゆ':'yu', 'よ':'yo', 'ら':'ra', 'り':'ri', 'る':'ru', 'れ':'re', 'ろ':'ro',
    'わ':'wa', 'を':'wo', 'ん':'n', 'が':'ga', 'ぎ':'gi', 'ぐ':'gu', 'げ':'ge', 'ご':'go',
    'ざ':'za', 'じ':'ji', 'ず':'zu', 'ぜ':'ze', 'ぞ':'zo', 'だ':'da', 'ぢ':'ji', 'づ':'zu', 'で':'de', 'ど':'do',
    'ば':'ba', 'び':'bi', 'ぶ':'bu', 'べ':'be', 'ぼ':'bo', 'ぱ':'pa', 'ぴ':'pi', 'ぷ':'pu', 'ぺ':'pe', 'ぽ':'po'
}

def hiragana_to_romaji(hira_str):
    if not hira_str: 
        return ""
    
    result = ""
    i = 0
    while i < len(hira_str):
        # 1. Check for 2-character combos first (e.g., きゃ -> kya)
        if i + 1 < len(hira_str) and hira_str[i:i+2] in HIRA_TO_ROMAJI:
            result += HIRA_TO_ROMAJI[hira_str[i:i+2]]
            i += 2
        # 2. Check for small tsu (double consonant)
        elif hira_str[i] == 'っ':
            if i + 1 < len(hira_str):
                next_char = HIRA_TO_ROMAJI.get(hira_str[i+1], "")
                if next_char:
                    # Double the first letter of the next character
                    result += next_char[0] 
            i += 1
        # 3. Standard single characters
        elif hira_str[i] in HIRA_TO_ROMAJI:
            result += HIRA_TO_ROMAJI[hira_str[i]]
            i += 1
        # 4. Fallback for unmapped chars (symbols, kanji, etc.)
        else:
            result += hira_str[i]
            i += 1
    return result

# STEP 6: Create word-level alignments using Awesome-Align, which is based on multilingual BERT and can handle many-to-many alignments.

def create_word_alignments(final_df):
    temp_dir = tempfile.gettempdir()
    
    data_file = os.path.join(temp_dir, "alignments_input.txt")
    output_file = os.path.join(temp_dir, "alignments_output.txt")

    with open(data_file, "w", encoding='utf8') as f:
        for src, tgt in zip(final_df["jp_final"], final_df["en_translation"]):
            f.write(f"{src} ||| {tgt}\n")
    
    docker_model_path = "/app/awesome-align/model_without_co"

    if os.path.exists(docker_model_path):
        actual_model_path = docker_model_path
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        actual_model_path = os.path.join(parent_dir, "awesome-align", "model_without_co")
        
        log_gcp("DEBUG", f"Running locally! Swapped model path to: {actual_model_path}")

    env = os.environ.copy()
    env["DATA_FILE"] = data_file
    env["MODEL_NAME_OR_PATH"] = actual_model_path
    env["OUTPUT_FILE"] = output_file

    subprocess.run(
        [
            "awesome-align",
            "--output_file", env["OUTPUT_FILE"],
            "--model_name_or_path", env["MODEL_NAME_OR_PATH"],
            "--data_file", env["DATA_FILE"],
            "--extraction", "softmax",
            "--batch_size", "32",
            "--num_workers", "0", 
        ],
        env=env,
        check=True
    )

    with open(output_file, "r", encoding='utf8') as f:
        alignments = f.readlines()
    
    alignments = [line.strip() for line in alignments]
    final_df["alignments"] = alignments

    essential_df = final_df[["jp_final", "en_translation", "ro_final", "alignments", "start_time", "end_time"]]
    essential_df = essential_df.rename(columns={
        "jp_final": "jp",
        "en_translation": "en",
        "ro_final": "romaji"
    })

    return essential_df

# STEP 7.1: Filter the alignments to only include content words, and remove any alignments that are based on common particles or auxiliary verbs that don't carry much meaning on their own.

GRAMMAR_LEMMAS = {
    'する', 'ある', 'いる', 'なる', 'くる', 'いく', 
    'こと', 'もの', 'よう', 'わけ', 'そう', 'ため', 
    'いい', 'よい', 'ない', '様', 'どうも'
}

def filter_alignments_pos(essential_df, verbose_debug=False):
    filtered_df = essential_df.copy()

    alignments_col = essential_df["alignments"]
    jp_col = essential_df["jp"]
    en_col = essential_df["en"]

    for idx, (a, j, e) in enumerate(zip(alignments_col, jp_col, en_col)):
        jp_words = j.split()
        en_words = e.split()
        alignments = a.split()
        filtered_alignments = ""

        for align in alignments:
            src_idx, tgt_idx = map(int, align.split("-"))
            
            if src_idx >= len(jp_words) or tgt_idx >= len(en_words):
                continue
                
            jp_word = jp_words[src_idx]
            en_word = en_words[tgt_idx]
            
            if jp_word.isascii() and jp_word.isalnum():
                if verbose_debug:
                    log_gcp("DEBUG", f"Filtered out alignment {align} because JP source '{jp_word}' is English/ASCII\nJP '{j}'\nEN '{e}'\n")
                continue

            tag = tagger(jp_word)[0]
            jp_pos1 = tag.feature.pos1
            jp_pos2 = tag.feature.pos2
            
            jp_lemma = getattr(tag.feature, 'lemma', None)
            if not jp_lemma or jp_lemma == '*':
                jp_lemma = jp_word

            if jp_pos1 in ["助詞", '代名詞', '接続詞', '助動詞','感動詞', '記号']:
                if verbose_debug:
                    log_gcp("DEBUG", f"Filtered out alignment {align} with word '{jp_word}' based on JP POS1 '{jp_pos1}'\nJP '{j}'\nEN '{e}'\n")
                continue

            if jp_pos2 in ["非自立", "数詞", "接尾"]:
                if verbose_debug:
                    log_gcp("DEBUG", f"Filtered out alignment {align} with word '{jp_word}' based on JP POS2 '{jp_pos2}'\nJP '{j}'\nEN '{e}'\n")
                continue

            if jp_lemma in GRAMMAR_LEMMAS or jp_word in GRAMMAR_LEMMAS:
                if verbose_debug:
                    log_gcp("DEBUG", f"Filtered out alignment {align} with word '{jp_word}' based on JP lemma '{jp_lemma}'\nJP '{j}'\nEN '{e}'\n")
                continue
            
            en_doc = nlp(en_word)
            if not en_doc:
                continue
            en_token = en_doc[0]
            en_pos = en_token.pos_

            if en_pos in ["AUX", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "PUNCT"]:
                if verbose_debug:
                    log_gcp("DEBUG", f"Filtered out alignment {align} with word '{en_word}' based on EN POS '{en_pos}'\nJP '{j}'\nEN '{e}'\n")
                continue

            if en_token.is_stop:
                if verbose_debug:
                    log_gcp("DEBUG", f"Filtered out alignment {align} with word '{en_word}' because EN word is a stopword\nJP '{j}'\nEN '{e}'\n")
                continue

            filtered_alignments += f"{src_idx}-{tgt_idx} "

        filtered_df.loc[idx, "alignments"] = filtered_alignments.strip()

    return filtered_df

# Step 7.2: Filter out any alignments using LaBSE similarity.

LABSE_MODEL = SentenceTransformer("sentence-transformers/LaBSE")

def _calculate_labse_similarity(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
        
    embeddings = LABSE_MODEL.encode([text_a, text_b], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    return similarity

def filter_alignments_labse(filtered_df, threshold=0.60, verbose_debug=False):

    def combine_bidirectional_overlaps(pairs):
        graph = defaultdict(set)
        nodes = set()

        for s, t in pairs:
            s_node = f"{s}_s"
            t_node = f"{t}_t"
            graph[s_node].add(t_node)
            graph[t_node].add(s_node)
            nodes.add(s_node)
            nodes.add(t_node)

        visited = set()
        groups = []

        def dfs(node, group):
            visited.add(node)
            group.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, group)

        for node in nodes:
            if node not in visited:
                group = []
                dfs(node, group)
                groups.append(group)

        return groups

    log_gcp("INFO", "Running LaBSE semantic filtering on alignment groups.")
    final_df = filtered_df.copy()

    for idx in final_df.index:
        j = final_df.at[idx, "jp"]
        e = final_df.at[idx, "en"]
        a = final_df.at[idx, "alignments"]

        if not str(a).strip():
            continue

        jp_words = j.split()
        en_words = e.split()
        alignments = a.split()

        pairs = []
        for align in alignments:
            src, tgt = map(int, align.split("-"))
            if src < len(jp_words) and tgt < len(en_words):
                pairs.append((src, tgt))

        groups = combine_bidirectional_overlaps(pairs)
        surviving_alignments = []

        for group in groups:
            s_indices = sorted([int(n.split('_')[0]) for n in group if n.endswith('_s')])
            t_indices = sorted([int(n.split('_')[0]) for n in group if n.endswith('_t')])

            jp_chunk = "".join([jp_words[i] for i in s_indices])
            en_chunk = " ".join([en_words[i] for i in t_indices])

            similarity = _calculate_labse_similarity(jp_chunk, en_chunk)

            if similarity >= threshold:
                for s in s_indices:
                    for t in t_indices:
                        if (s, t) in pairs:
                            surviving_alignments.append(f"{s}-{t}")
            else:
                if verbose_debug:
                    log_gcp("DEBUG", f"Dropped group (Low Semantics): JP '{jp_chunk}' -> EN '{en_chunk}' (Score: {similarity:.2f})\nJP '{j}'\nEN '{e}'\n")

        final_df.at[idx, "alignments"] = " ".join(surviving_alignments)

    return final_df

# Step 7.3: Filter out any alignments using LLMs. If it takes too long or failes, simply restore the previous data.

def _call_openai(prompt, model, timeout_sec):
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout_sec,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except APITimeoutError as e:
        raise TimeoutError(f"OpenAI Timeout: {e}")
    except Exception as e:
        raise RuntimeError(f"OpenAI Error: {e}")


def _call_gemini(prompt, model, timeout_sec):
    timeout_ms = int(timeout_sec * 1000)
    
    try:
        client = genai.Client(
            http_options=types.HttpOptions(timeout=timeout_ms)
        )
    except Exception as e:
        raise ValueError(f"Failed to initialize Gemini. Is GEMINI_API_KEY set? {e}")
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            )
        )
        return response.text.strip()
        
    except Exception as e:
        raise RuntimeError(f"Gemini API Error: {e}")

def filter_alignments_llm(filtered_df, model="gpt-4o-mini", batch_size=20, conservative=False, timeout_sec=30.0, verbose_debug=False):
    log_gcp("INFO", f"Running unified LLM alignment filtering using {model}.")
    
    df_copy = filtered_df.copy()

    for batch_start in range(0, len(df_copy), batch_size):
        log_gcp("DEBUG", f"Processing LLM batch starting at index {batch_start}...")
            
        batch = df_copy.iloc[batch_start:batch_start + batch_size]

        c = "Adopt a conservative bias in your evaluation. You have the discretion to remove 'free' or loose alignments that do not maintain clear and direct semantic logic." if conservative else ""

        prompt = f"""I am going to give you a batch of Japanese sentences and their corresponding English sentences that mean the same thing. 
I am also going to give you candidate word alignments for each sentence pair that may or may not be correct. 
Your task is to tell me for each alignment whether it is correct, separated by spaces on a single line for each sentence. Do not output anything else.
{c}

Example batch:

Japanese: そう だ 明日 は 海 へ 行こう
English: Yes , let’s go to the sea tomorrow
Proposed Alignments:
4-6 (海 - sea)
6-3 (行こう - go)
6-7 (行こう - tomorrow)

Japanese: 私 は 昨日 本 を 読んだ
English: I read a book yesterday
Proposed Alignments:
1-0 (私 - I)
3-2 (本 - book)
4-3 (読んだ - read)

Your output:
1 1 0
1 1 1

Now check the following sentences:
"""

        row_map = {}
        for idx, row in batch.iterrows():
            jp_sentence = row["jp"]
            en_sentence = row["en"]
            alignments = row["alignments"].split()
            
            if not alignments:
                row_map[idx] = 0
                continue
                
            row_map[idx] = len(alignments)
            jp_words = jp_sentence.split()
            en_words = en_sentence.split()
            prompt += f"\nJapanese: {jp_sentence}\nEnglish: {en_sentence}\nProposed Alignments:\n"
            
            for a in alignments:
                src_idx, tgt_idx = map(int, a.split("-"))
                jp_word = jp_words[src_idx] if src_idx < len(jp_words) else "?"
                en_word = en_words[tgt_idx] if tgt_idx < len(en_words) else "?"
                prompt += f"{a} ({jp_word} - {en_word})\n"
            
        if verbose_debug:
            log_gcp("DEBUG", f"Constructed LLM Prompt for batch starting at index {batch_start}:\n{prompt}\n---")
            
        try:
            if "gpt" in model.lower() or "o1" in model.lower():
                output_text = _call_openai(prompt, model, timeout_sec)
            elif "gemini" in model.lower():
                output_text = _call_gemini(prompt, model, timeout_sec)
            else:
                log_gcp("WARNING", f"Model '{model}' not recognized by router. Skipping batch.")
                continue
                
        except (TimeoutError, RuntimeError, ValueError) as e:
            log_gcp("WARNING", f"{e}. Aborting LLM filter for this batch and keeping original alignments.")
            continue

        if verbose_debug:
            log_gcp("DEBUG", f"LLM Output for batch starting at index {batch_start}:\n{output_text}\n---")

        output_lines = [line.strip() for line in output_text.splitlines() if line.strip()]

        current_line = 0
        for idx, row in batch.iterrows():
            n = row_map[idx]
            if n == 0:
                continue
                
            try:
                alignments = row["alignments"].split()
                row_output = output_lines[current_line].split() if current_line < len(output_lines) else ["1"] * n
                
                while len(row_output) < n:
                    row_output.append("1")
                    
                filtered = []
                dropped = []
                
                for align, keep in zip(alignments, row_output):
                    if keep == "1":
                        filtered.append(align)
                    else:
                        dropped.append(align)
                
                if verbose_debug and dropped:
                    jp_words = row["jp"].split()
                    en_words = row["en"].split()
                    dropped_text = [f"({jp_words[int(d.split('-')[ 0 ])] if int(d.split('-')[ 0 ]) < len(jp_words) else '?'} -> {en_words[int(d.split('-')[ 1 ])] if int(d.split('-')[ 1 ]) < len(en_words) else '?'})" for d in dropped]
                    log_gcp("DEBUG", f"LLM Dropped: {', '.join(dropped_text)}\nJP '{row['jp']}' // EN '{row['en']}'")

                df_copy.loc[idx, "alignments"] = " ".join(filtered)
                
            except Exception as e:
                log_gcp("WARNING", f"Failed to parse LLM output for row {idx}... {e}. Keeping original alignments.")
            finally:
                current_line += 1

    return df_copy

def filter_alignments_llms(filtered_df, models=["gpt-4o-mini", "gemini-3.5-flash"], verbose_debug=False):
    working_df = filtered_df.copy()
    
    for m in models:
        checkpoint_df = working_df.copy()
        try:
            working_df = filter_alignments_llm(working_df, model=m, verbose_debug=verbose_debug)
        except Exception as e:
            log_gcp("ERROR", f"LLM filtering failed entirely with model {m}... {e}. Restoring previous alignments and continuing.")
            working_df = checkpoint_df
            continue

    return working_df

# Step 8: Filter out any alignments that don't have a valid definition, while extracting the definitions for the ones that do.

def extract_vocabulary(filtered_df):
    log_gcp("INFO", "Extracting vocabulary and fetching JMdict definitions.")

    vocab_map = []
    word_cache = {}
    seen_uids = set()

    for idx in filtered_df.index:
        j = filtered_df.at[idx, "jp"]
        r = filtered_df.at[idx, "romaji"]
        a = filtered_df.at[idx, "alignments"]
        
        if not a or str(a).strip() == "":
            continue

        jp_words = j.split()
        ro_words = r.split()
        alignments = a.split()

        surviving_alignments = []

        for alignment in alignments:
            src_idx_str, _ = alignment.split("-")
            src_idx = int(src_idx_str)

            if src_idx >= len(jp_words):
                log_gcp("ERROR", f"Source index {src_idx} out of bounds for {len(jp_words)} JP words. Skipping this alignment...")
                continue
        
            jp_word = jp_words[src_idx]
    
            if src_idx >= len(ro_words):
                log_gcp("ERROR", f"Source index {src_idx} out of bounds for {len(ro_words)} Romaji words. Skipping this alignment...")
                continue

            ro_word = ro_words[src_idx]

            word_tags = tagger(jp_word)
            if not word_tags:
                log_gcp("ERROR", f"Failed to parse JP word '{jp_word}' with fugashi. Skipping this word...")
                continue

            if len(word_tags) > 1:
                continue

            tag = word_tags[0]

            base_form = getattr(tag.feature, 'lemma', None)
            if not base_form or base_form == '*':
                base_form = jp_word
                
            if base_form in word_cache:
                cached_uid = word_cache[base_form]
                if cached_uid is not False:
                    surviving_alignments.append(f"{alignment}:{cached_uid}")
                else:
                    continue

            result = jam.lookup(base_form, exact_match=True)

            if result.entries:
                selected_entry = result.entries[0]

                if len(result.entries) > 1 and ro_word:
                    clean_target_romaji = ro_word.lower()

                    for entry in result.entries:
                        entry_romaji_readings = [hiragana_to_romaji(k.text) for k in entry.kana_forms]

                        if clean_target_romaji in entry_romaji_readings:
                            selected_entry = entry
                            break

                uid = selected_entry.idseq
                surviving_alignments.append(f"{alignment}:{uid}")
                word_cache[base_form] = uid

                if uid in seen_uids:
                    continue

                readings = []
                if selected_entry.kana_forms:
                    for reading in selected_entry.kana_forms:
                        readings.append({
                            "text": reading.text,
                            "pri": reading.pri,
                            "info": reading.info
                        })

                writings = []
                if selected_entry.kanji_forms:
                    for writing in selected_entry.kanji_forms:
                        writings.append({
                            "text": writing.text,
                            "pri": writing.pri,
                            "info": writing.info 
                        })

                senses = []
                for sense in selected_entry.senses:
                    senses.append({
                        "meanings": sense.text().replace("/", "; "), 
                        "parts_of_speech": list(sense.pos),
                        "misc_notes": list(sense.misc),
                        "cross_references": list(sense.xref),
                        "antonyms": list(sense.antonym),
                        "fields": list(sense.field),
                        "dialects": list(sense.dialect),
                        "extra_info": list(sense.info)
                    })
                
                vocab_map.append({
                    "uid": uid,
                    "readings": readings,
                    "writings": writings,
                    "senses": senses, 
                    "first_seen_line": f"{idx}:{alignment}"
                })

                seen_uids.add(uid)
                
            else:
                word_cache[base_form] = False
        
        filtered_df.at[idx, "alignments"] = " ".join(surviving_alignments)
    
    log_gcp("INFO", f"Successfully extracted {len(vocab_map)} aligned vocab words with definitions.")
    return filtered_df, vocab_map

def run_alignment_pipeline(clean_df, filter_llms=["gpt-4o-mini"], verbose_pos=False, verbose_labse=False, verbose_llm=False, progress_cb=None):
    try:
        if progress_cb: progress_cb(ProcessingStep.ALIGNING_TOKENS)
        aligned_df = create_word_alignments(clean_df)
    except Exception as e:
        log_gcp("ERROR", f"Failed to create word alignments... {e}", traceback=traceback.format_exc())
        raise AlignmentError("Failed to create word alignments...", "ERR_ALIGNMENT_FAILED")
    
    try:
        if progress_cb: progress_cb(ProcessingStep.FILTERING_POS)
        filtered_df = filter_alignments_pos(aligned_df, verbose_debug=verbose_pos)
    except Exception as e:
        log_gcp("ERROR", f"Failed to filter alignments using Parts of Speech... {e}", traceback=traceback.format_exc())
        raise AlignmentError("Failed to filter alignments using Parts of Speech...", "ERR_POS_FILTERING_FAILED")
    
    try:
        if progress_cb: progress_cb(ProcessingStep.FILTERING_LABSE)
        filtered_df = filter_alignments_labse(filtered_df, verbose_debug=verbose_labse)
    except Exception as e:
        log_gcp("ERROR", f"Failed to filter alignments using LaBSE... {e}", traceback=traceback.format_exc())
        raise AlignmentError("Failed to filter alignments using LaBSE...", "ERR_LABSE_FILTERING_FAILED")
    
    try:
        if progress_cb: progress_cb(ProcessingStep.FILTERING_LLM)
        filtered_df = filter_alignments_llms(filtered_df, models=filter_llms, verbose_debug=verbose_llm)
    except Exception as e:
        log_gcp("ERROR", f"Failed to filter alignments using LLMs... {e}", traceback=traceback.format_exc())
        raise AlignmentError("Failed to filter alignments using LLMs...", "ERR_LLM_FILTERING_FAILED")

    try:
        if progress_cb: progress_cb(ProcessingStep.EXTRACTING_VOCAB)
        final_df, vocab_map = extract_vocabulary(filtered_df)
    except Exception as e:
        log_gcp("ERROR", f"Failed to filter alignments using definitions... {e}", traceback=traceback.format_exc())
        raise AlignmentError("Failed to filter alignments using Parts of Speech...", "ERR_VOCAB_FILTERING_FAILED")

    if len(vocab_map) < 5:
        log_gcp("WARNING", "All translation sources failed to yield a matching result...", query=vocab_map)
        raise AlignmentError("Song didn't have enough vocab...", "EXPT_NOT_ENOUGH_VOCAB")
    elif len(vocab_map) > 15:
        vocab_map = random.sample(vocab_map, 15)
        vocab_uids = [vocab["uid"] for vocab in vocab_map]

        for idx in final_df.index:
            a = final_df.at[idx, "alignments"]
            
            if not a or str(a).strip() == "":
                continue

            alignments = a.split()
            surviving_alignments = []

            for alignment in alignments:
                uid = int(alignment.split(":")[1])
                if uid in vocab_uids:
                    surviving_alignments.append(alignment)
            
            final_df.at[idx, "alignments"] = " ".join(surviving_alignments)

    return final_df, vocab_map
