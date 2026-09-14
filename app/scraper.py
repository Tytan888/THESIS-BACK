import os
import pandas as pd
import re
import sys
import traceback
import time
from ytmusicapi import YTMusic
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from urllib.parse import quote
from pyvirtualdisplay import Display
import unicodedata
from ddgs import DDGS

from logger import log_gcp

class ScraperError(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code

def _is_match_above_threshold(s1: str, s2: str, threshold: float = 80.0) -> bool:
    s1 = unicodedata.normalize('NFKC', s1)
    s2 = unicodedata.normalize('NFKC', s2)
    s1 = s1.lower()
    s2 = s2.lower()
    s1 = s1.replace(" ", "").replace("\n", "")
    s2 = s2.replace(" ", "").replace("\n", "")
    similarity = SequenceMatcher(None, s1, s2).ratio() * 100
    return similarity >= threshold

def _get_match_value(s1: str, s2: str) -> float:
    s1 = unicodedata.normalize('NFKC', s1)
    s2 = unicodedata.normalize('NFKC', s2)
    s1 = s1.lower()
    s2 = s2.lower()
    s1 = s1.replace(" ", "").replace("\n", "")
    s2 = s2.replace(" ", "").replace("\n", "")
    similarity = SequenceMatcher(None, s1, s2).ratio() * 100
    return similarity

def _get_headless_driver():
    display = None
    if sys.platform.startswith('linux'):
        display = Display(visible=0, size=(1920, 1080))
        display.start()

    options = webdriver.ChromeOptions()
    options.page_load_strategy = 'none' 
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    
    if sys.platform.startswith('linux'):
        options.binary_location = "/usr/bin/chromium"

    driver = webdriver.Chrome(options=options)
    
    return driver, display

# STEP 1: Get the lyrics using YouTube Music API. If not found, use SyncedLyrics API. If still not found, return None.

def scrape_baseline_youtube(video_id: str):
    location = "PH"
    language = "en"

    log_gcp("INFO", f"Searching YouTube Music", video_id=video_id)

    raw_proxies = [
        os.environ.get("JP_PROXY_URL"),
        os.environ.get("PH_PROXY_URL_1"),
        os.environ.get("PH_PROXY_URL_2")
    ]
    proxies_list = [{"http": p, "https": p} for p in raw_proxies if p]
    proxies_list.append(None)

    error_thrown = False

    for proxies in proxies_list:
        if proxies:
            log_gcp("INFO", "Proxy configuration loaded.", video_id=video_id, proxy_url=proxies["http"])
        else:
            log_gcp("WARNING", "No proxy configuration found. Proceeding without proxy.", video_id=video_id)
    
        ytmusic = YTMusic(location=location, language=language, proxies=proxies)

        try:
            try:
                watch_playlist = ytmusic.get_watch_playlist(videoId=video_id)
            except Exception as e:
                log_gcp("WARNING", f"Song could not be accessed. May be an invalid video_id...", video_id=video_id, location=location, language=language, traceback=traceback.format_exc())
                # raise ScraperError("Song could not be accessed. May be an invalid video_id...", "ERR_YT_NOT_FOUND")
                continue

            track_info = watch_playlist['tracks'][0]
            title = track_info.get('title', 'Unknown Title')

            artists_data = track_info.get('artists', [])
            artists = ", ".join([a['name'] for a in artists_data]) if artists_data else "Unknown Artist"

            log_gcp("DEBUG", f"Attempting to pull lyrics for '{title}' by {artists}", video_id=video_id)

            lyrics_id = watch_playlist.get('lyrics')
            if not lyrics_id:
                log_gcp("WARNING", "No lyrics returned for this song...", video_id=video_id)
                # raise ScraperError("No lyrics returned for this song...", "EXPT_YT_NO_LYRICS")
                continue
            
            try:
                lyrics_data = ytmusic.get_lyrics(lyrics_id, timestamps=True)
            except Exception as e:
                log_gcp("WARNING", f"Lyrics found, but they are not time-synced...", video_id=video_id, traceback=traceback.format_exc())
                # raise ScraperError("Lyrics found, but they are not time-synced...", "EXPT_YT_NOT_SYNCED")
                continue

            if not lyrics_data:
                log_gcp("WARNING", "No lyrics data returned for this song...", video_id=video_id)
                # raise ScraperError("No lyrics returned for this song...", "EXPT_YT_NO_LYRICS")
                continue

            df = pd.DataFrame(columns=['jp_baseline', 'start_time', 'end_time'])
            for line in lyrics_data.get('lyrics', []):
                df.loc[len(df)] = [line.text, line.start_time, line.end_time]

            log_gcp("INFO", "Successfully generated YT Music baseline.", video_id=video_id)
            return df, title, artists

        except Exception as e:
            log_gcp("ERROR", f"YT Music extraction failed... {e}", traceback=traceback.format_exc(), video_id=video_id)
            # raise ScraperError("YT Music extraction failed...", "ERR_YT_API_GENERIC_CRASH")
            error_thrown = True
            continue

    if error_thrown:
        raise ScraperError("YT Music extraction failed...", "ERR_YT_API_GENERIC_CRASH")
    else:
        raise ScraperError("Lyrics couldn't be found or weren't time-synced.", "EXPT_YT_BAD_LYRICS")
    
# --- STEP 2: TRANSLATION SCRAPERS ---

def _is_valid_umatime_url(url: str) -> bool:
    prefix = "https://www.utatime.com/global/lyrics/"
    if not url.startswith(prefix):
        return False
    rest = url[len(prefix):].rstrip("/") 
    return rest.count("/") == 1

def _scrape_utatime(title: str, artists: str, baseline_text: str):
    
    def _gather_utatime_links(search_query: str):
        driver, display = _get_headless_driver()
        links = []
        log_gcp("INFO", "Attempting UtaTime scrape...", query=search_query)

        try:
            driver.get("https://www.utatime.com/global/lyrics/#search")
            wait = WebDriverWait(driver, 10)
            
            try:
                search_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.search-input")))
                search_box.send_keys(search_query)
            except TimeoutException:
                log_gcp("WARNING", "Timeout: Search box ('input.search-input') not found. Site UI may have changed.")
                return None

            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".search-results-meta")))
            except TimeoutException:
                log_gcp("WARNING", "Timeout: Search results ('.search-result') failed to load. Possibly no results or blocked.")
                return None
            
            soup = BeautifulSoup(driver.page_source, "html.parser")
            
            for r in soup.select(".search-result-lyric.search-result"):
                if _is_valid_umatime_url(r.get("href", "")):
                    links.append(r.get("href"))
                    
            return links
        except Exception as e:
            log_gcp("ERROR", f"UtaTime search phase failed... {e}", traceback=traceback.format_exc())
            return None
        finally:
            if driver:
                driver.quit()
            if display:
                display.stop()

    links = []
    links = _gather_utatime_links(f"{title} {artists}")
    
    if not links:
        links = _gather_utatime_links(f"{title}")

    if not links:
        log_gcp("WARNING", "No search results found on Lyrical Nonsense.")
        return None, None

    for link in links[:3]:
        log_gcp("DEBUG", f"Extracting from: {link}")
        
        ext_driver, ext_display = _get_headless_driver()
        
        try:
            ext_driver.get(link)
            wait = WebDriverWait(ext_driver, 10)
            
            try:
                wait.until(EC.presence_of_element_located((By.ID, "Romaji")))
            except TimeoutException:
                log_gcp("WARNING", f"Timeout: 'Romaji' div not found on {link}. Skipping to next link.")
                continue

            soup = BeautifulSoup(ext_driver.page_source, "html.parser")
            
            def get_lines(div_id):
                div = soup.find("div", id=lambda x: x and div_id in x)

                if not div:
                    return []
                
                for rt_span in div.find_all("span", class_="rt"):
                    rt_span.decompose()

                return [span.get_text(strip=True) for span in div.find_all("span", class_="line-text")] if div else []

            ro_lines = get_lines("Romaji")
            jp_lines = get_lines("Original")
            en_lines = get_lines("English") or get_lines("EnglishOff")

            if not (ro_lines and jp_lines and en_lines):
                log_gcp("DEBUG", "One of the translations was missing, skipping this result.")
                continue
            if not _is_match_above_threshold("".join(jp_lines), "".join(baseline_text), 80):
                log_gcp("DEBUG", f"Baseline text did not match well with the Japanese translation ({_get_match_value(''.join(jp_lines), ''.join(baseline_text)):.2f}%), skipping this result.")
                continue
            if len(jp_lines) != len(en_lines) or len(jp_lines) != len(ro_lines):
                    log_gcp("DEBUG", f"Line count mismatch (JP: {len(jp_lines)} | EN: {len(en_lines)} | RO: {len(ro_lines)}). Skipping.")
                    continue

            log_gcp("INFO", "Successfully extracted UtaTime translations.")
            return pd.DataFrame({'jp_translation': jp_lines, 'en_translation': en_lines, 'ro_translation': ro_lines}), link

        except Exception as e:
            log_gcp("WARNING", f"Failed extracting {link}... {e}")
            continue 
            
        finally:
            if ext_driver:
                ext_driver.quit()
            if ext_display:
                ext_display.stop()

    # If the loop finishes and none passed the tests:
    log_gcp("WARNING", "All search results on UtaTime failed verification.")
    return None, None

def _scrape_lyrics_translate(search_query: str, baseline_text: str):
    log_gcp("INFO", "Attempting Lyrics Translate scrape...", query=search_query)
    driver, display = _get_headless_driver()
    
    try:
        encoded_search = quote(search_query)
        driver.get(f"https://lyricstranslate.com/en/site-search?query={encoded_search}")

        wait = WebDriverWait(driver, 10)
        
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.search-res__body")))
        except TimeoutException:
            log_gcp("WARNING", "Timeout: Search results container not found. Site UI may have changed.")
            return None, None

        soup = BeautifulSoup(driver.page_source, "html.parser")
        songs_div = soup.find("div", class_="_songs")

        links = []
        if songs_div:
            items = songs_div.find_all("div", class_="block-search-res__item")
            
            for item in items:
                title_link = item.find("a", class_="block-1-table__title")
                
                if title_link and title_link.get("href"):
                    href = title_link["href"]
                    if href.startswith("/"):
                        href = "https://lyricstranslate.com" + href
                    
                    if href not in links:
                        links.append(href)

        if not links:
            log_gcp("WARNING", "No search results found on Lyrics Translate.")
            return None, None

        for link in links[:3]:
            log_gcp("DEBUG", f"Extracting from: {link}")
            driver.get(link)

            try:
                wait.until(EC.presence_of_element_located((By.ID, "song-body")))
            except TimeoutException:
                log_gcp("WARNING", f"Timeout: 'song-body' div not found on {link}. Skipping to next link.")
                continue
            
            try:
                translit_toggle = wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "a.js-song-translit")
                ))
                
                driver.execute_script("arguments[ 0 ].scrollIntoView({block: 'center'});", translit_toggle)
                time.sleep(0.5) 
                
                translit_toggle.click()
                time.sleep(0.5)
                
            except TimeoutException:
                log_gcp("WARNING", f"Timeout: Transliteration toggle button not found/clickable on {link}. Skipping.")
                continue
            except Exception as e:
                log_gcp("WARNING", f"Failed to click transliteration toggle... {e}")
                continue
                
            try:
                translit_option = wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, 'div[data-more-content="transliterations_overlay"] a.js-translit-link')
                ))
                
                driver.execute_script("arguments[ 0 ].click();", translit_option)
            except TimeoutException:
                log_gcp("WARNING", f"Timeout: Romanization option not found in dropdown on {link}. Skipping.")
                continue

            try:
                wait.until(EC.presence_of_element_located((By.ID, "song-transliteration")))
                wait.until(lambda d: d.find_element(By.ID, "song-transliteration").text.strip() != "")
            except TimeoutException:
                log_gcp("WARNING", f"Timeout: Transliteration text failed to load/populate on {link}. Skipping.")
                continue

            soup = BeautifulSoup(driver.page_source, "html.parser")

            def get_lines(div_id):
                song_body = soup.find("div", id=div_id)
                lines_list = []
                if song_body:
                    divs = song_body.find_all("div", class_=re.compile(r"ll-\d+-\d+"))
                    last_x = None
                    for div in divs:
                        match = re.match(r"ll-(\d+)-(\d+)", div["class"][ 0 ])
                        if match:
                            x = int(match.group(1))
                            if last_x is not None and x != last_x:
                                lines_list.append("")
                            lines_list.append(div.get_text(strip=True))
                            last_x = x
                return lines_list

            ro_lines = get_lines("song-transliteration")
            jp_lines = get_lines("song-body")
            
            # Find English translation link
            en_lines = []
            eng_link = None
            for h in soup.find_all("h3", class_="title"):
                if h.get_text(strip=True).lower() == "translations":
                    sibling = h.find_next_sibling("div", class_="tags__items")
                    while sibling:
                        if "english" in sibling.get_text(strip=True).lower():
                            a_tag = sibling.find("a")
                            if a_tag and a_tag.get("href"):
                                eng_link = "https://lyricstranslate.com" + a_tag["href"] if a_tag["href"].startswith("/") else None
                                break
                        sibling = sibling.find_next_sibling("div", class_="slist")
                    break 

            if eng_link:
                driver.get(eng_link)
                try:
                    wait.until(EC.presence_of_element_located((By.ID, "translation-body")))
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    en_lines = get_lines("translation-body")
                except TimeoutException:
                    log_gcp("WARNING", f"Timeout: 'translation-body' failed to load on English page for {link}. Skipping.")
                    continue

            if not (ro_lines and jp_lines and en_lines):
                log_gcp("DEBUG", "Missing translation columns (ro, jp, or en). Skipping to next link.")
                continue
            if not _is_match_above_threshold("".join(jp_lines), "".join(baseline_text), 80):
                log_gcp("DEBUG", f"Baseline text did not match well with the Japanese translation ({_get_match_value(''.join(jp_lines), ''.join(baseline_text)):.2f}%), skipping this result.")
                continue
            if len(jp_lines) != len(en_lines) or len(jp_lines) != len(ro_lines):
                    log_gcp("DEBUG", f"Line count mismatch (JP: {len(jp_lines)} | EN: {len(en_lines)} | RO: {len(ro_lines)}). Skipping.")
                    continue

            log_gcp("INFO", "Successfully extracted Lyrics Translate translations.")
            return pd.DataFrame(
                {'jp_translation': jp_lines, 'en_translation': en_lines, 'ro_translation': ro_lines}
            ).fillna(""), link
        
        log_gcp("WARNING", "All Lyrics Translate links failed validation checks.")
        return None, None

    except Exception as e:
        log_gcp("ERROR", f"Lyrics Translate scrape failed... {e}", traceback=traceback.format_exc())
        return None, None
    finally:
        if driver:
            driver.quit()
        if display:
            display.stop()

def _scrape_animeliryca(title: str, artists: str, baseline_text: str):
    
    def _gather_animeliryca_links(query_string: str, is_ddgs: bool = False):
        if is_ddgs:
            log_gcp("INFO", "Attempting Animeliryca scrape via DDGS fallback...", query=query_string)
            try:
                results = DDGS().text(f"{query_string} site:animeliryca.com", max_results=10, backend='google')
                links = [r['href'] for r in results if 'animeliryca.com' in r['href']]
                return links
            except Exception as e:
                log_gcp("WARNING", f"DDGS search failed... {e}")
                return []

        log_gcp("INFO", "Attempting Animeliryca site search...", query=query_string)
        driver, display = _get_headless_driver()
        links = []
        try:
            encoded_search = quote(query_string)
            driver.get(f"https://animeliryca.com/?s={encoded_search}")
            
            wait = WebDriverWait(driver, 10)
            
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.entry-thumb-full, section.no-results")))
            except TimeoutException:
                log_gcp("WARNING", "Timeout: Search results failed to load on Animeliryca.")
                return []

            soup = BeautifulSoup(driver.page_source, "html.parser")
            
            for div in soup.find_all("div", class_="entry-thumb-full"):
                a_tag = div.find("a", href=True)
                if a_tag:
                    links.append(a_tag["href"])
            
            return links
            
        except Exception as e:
            log_gcp("ERROR", f"Animeliryca search phase failed... {e}", traceback=traceback.format_exc())
            return []
        finally:
            if driver:
                driver.quit()
            if display:
                display.stop()

    links = _gather_animeliryca_links(f"{title} {artists}")
    
    if not links:
        log_gcp("INFO", "No results for Title + Artist, falling back to Title only.")
        links = _gather_animeliryca_links(f"{title}")
        
    if not links:
        log_gcp("INFO", "No results for Title only, falling back to DDGS.")
        links = _gather_animeliryca_links(f"{title} {artists}", is_ddgs=True)

    if not links:
        log_gcp("WARNING", "All search fallback methods failed for Animeliryca.")
        return None, None

    for link in links[:3]:
        log_gcp("DEBUG", f"Extracting from: {link}")
        
        ext_driver, ext_display = _get_headless_driver()
        
        try:
            ext_driver.get(link)
            wait = WebDriverWait(ext_driver, 10)
            
            try:
                wait.until(EC.presence_of_element_located((By.ID, "tablist1-panel1")))
            except TimeoutException:
                log_gcp("WARNING", f"Timeout: 'tablist1-panel1' div not found on {link}. Skipping to next link.")
                continue

            soup = BeautifulSoup(ext_driver.page_source, "html.parser")
            
            def get_lines(div_id):
                div = soup.find("div", id=div_id)
                if not div:
                    return []
                
                for blue_span in div.find_all("span", style=lambda value: value and "#0000ff" in value.lower()):
                    blue_span.decompose()
                
                raw_text = div.get_text(separator="\n", strip=True)
                
                return [
                    line.strip() 
                    for line in raw_text.split('\n') 
                    if line.strip() and not line.strip().lower().startswith("english translation")
                ]

            ro_lines = get_lines("tablist1-panel1")
            jp_lines = get_lines("tablist1-panel2")
            en_lines = get_lines("tablist1-panel4")

            if not (ro_lines and jp_lines and en_lines):
                log_gcp("DEBUG", "One of the translations was missing (ro, jp, or en), skipping this result.")
                continue
            
            if not _is_match_above_threshold("".join(jp_lines), "".join(baseline_text), 80):
                match_val = _get_match_value("".join(jp_lines), "".join(baseline_text))
                log_gcp("DEBUG", f"Baseline text did not match well with the Japanese translation ({match_val:.2f}%), skipping.")
                continue
                
            if len(jp_lines) != len(en_lines) or len(jp_lines) != len(ro_lines):
                    log_gcp("DEBUG", f"Line count mismatch (JP: {len(jp_lines)} | EN: {len(en_lines)} | RO: {len(ro_lines)}). Skipping.")
                    continue

            log_gcp("INFO", "Successfully extracted Animeliryca translations.")
            return pd.DataFrame({'jp_translation': jp_lines, 'en_translation': en_lines, 'ro_translation': ro_lines}), link

        except Exception as e:
            log_gcp("WARNING", f"Failed extracting {link}... {e}")
            continue 
            
        finally:
            if ext_driver:
                ext_driver.quit()
            if ext_display:
                ext_display.stop()

    log_gcp("WARNING", "All search results on Animeliryca failed verification.")
    return None, None

def _scrape_anime_song_lyrics(title: str, artists: str, baseline_text: str):
    
    def _gather_asl_links(query: str, search_type: str):
        links = []
        if search_type == "ddgs":
            log_gcp("INFO", "Attempting AnimeSongLyrics search via DDGS...", query=query)
            try:
                results = DDGS().text(f"{query} site:animesonglyrics.com", max_results=10, backend='google')
                for r in results:
                    if 'animesonglyrics.com' in r['href']:
                        links.append(r['href'])
            except Exception as e:
                log_gcp("WARNING", f"DDGS search failed... {e}")
                
        elif search_type == "native":
            log_gcp("INFO", "Attempting AnimeSongLyrics native site search...", query=query)
            driver, display = _get_headless_driver()
            try:
                encoded_search = quote(query)
                driver.get(f"https://animesonglyrics.com/results?q={encoded_search}")
                wait = WebDriverWait(driver, 10)
                
                try:
                    wait.until(EC.presence_of_element_located((By.ID, "songlist")))
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    songlist = soup.find("div", id="songlist")
                    
                    if songlist:
                        for a in songlist.find_all("a", href=True):
                            if "animesonglyrics.com" in a["href"] or a["href"].startswith("/"):
                                full_link = a["href"] if a["href"].startswith("http") else f"https://animesonglyrics.com{a['href']}"
                                links.append(full_link)
                                
                except TimeoutException:
                    log_gcp("WARNING", "Timeout: 'songlist' not found on AnimeSongLyrics results page.")
                    
            except Exception as e:
                log_gcp("ERROR", f"AnimeSongLyrics native search failed... {e}", traceback=traceback.format_exc())
            finally:
                if driver:
                    driver.quit()
                if display:
                    display.stop()
                    
        return list(dict.fromkeys(links))

    search_tiers = [
        (f"{title} {artists}", "ddgs"),
        (f"{title}", "ddgs"),
        (f"{title} {artists}", "native"),
        (f"{title}", "native")
    ]
    
    for query, search_type in search_tiers:
        links = _gather_asl_links(query, search_type)
        
        if not links:
            log_gcp("DEBUG", f"No links found for tier: {search_type} | {query}. Moving to next tier.")
            continue
            
        for link in links[:3]:
            log_gcp("DEBUG", f"Extracting from: {link}")
            
            ext_driver, ext_display = _get_headless_driver()
            try:
                ext_driver.get(link)
                wait = WebDriverWait(ext_driver, 10)
                
                try:
                    wait.until(EC.presence_of_element_located((By.ID, "tablyrics")))
                except TimeoutException:
                    log_gcp("WARNING", f"Timeout: 'tablyrics' div not found on {link}. Skipping.")
                    continue

                soup = BeautifulSoup(ext_driver.page_source, "html.parser")
                
                def get_lines(div_id_fragment):
                    divs_container = soup.find_all("div", class_=lambda x: x and div_id_fragment in x)
                    if not divs_container:
                        return []

                    container = divs_container[ 0 ]
                    
                    for rt_tag in container.find_all("rt"):
                        rt_tag.decompose()
                        
                    line_divs = container.find_all("div", class_="lineNormal")
                    lines = []

                    for line in line_divs:
                        children = [child for child in line.children if str(child).strip()]
                        
                        if len(children) == 1 and getattr(children[ 0 ], "name", None) == "hr":
                            lines = []
                            continue

                        text = line.get_text(strip=True)
                        if not text:
                            continue
                            
                        lower_text = text.lower()

                        ignore_phrases = (
                            "animesonglyrics", 
                            "アニメソングリリックス"
                        )
                        if any(phrase in lower_text for phrase in ignore_phrases):
                            continue
                            
                        if ("full version" in lower_text and "full version continues" not in lower_text) or ("fullバージョン" in lower_text and "fullバージョンのみ" not in lower_text):
                            lines = []
                            continue

                        if text.startswith('[') and text.endswith(']'):
                            continue

                        lines.append(text)

                    return lines

                ro_lines = get_lines("romajilyrics")
                jp_lines = get_lines("kanjilyrics")
                en_lines = get_lines("englishlyrics")

                if not (ro_lines and jp_lines and en_lines):
                    log_gcp("DEBUG", "Missing translation columns (ro, jp, or en). Skipping to next link.")
                    continue
                
                if not _is_match_above_threshold("".join(jp_lines), "".join(baseline_text), 80):
                    match_val = _get_match_value("".join(jp_lines), "".join(baseline_text))
                    log_gcp("DEBUG", f"Baseline text did not match well ({match_val:.2f}%). Skipping.")
                    continue
                    
                if len(jp_lines) != len(en_lines) or len(jp_lines) != len(ro_lines):
                    log_gcp("DEBUG", f"Line count mismatch (JP: {len(jp_lines)} | EN: {len(en_lines)} | RO: {len(ro_lines)}). Skipping.")
                    continue

                log_gcp("INFO", "Successfully extracted AnimeSongLyrics translations.")
                return pd.DataFrame({'jp_translation': jp_lines, 'en_translation': en_lines, 'ro_translation': ro_lines}), link

            except Exception as e:
                log_gcp("WARNING", f"Failed extracting {link}... {e}")
                continue 
            finally:
                if ext_driver:
                    ext_driver.quit()
                if ext_display:
                    ext_display.stop()
                    
        log_gcp("INFO", f"All links in tier '{search_type} | {query}' failed validation. Proceeding to next search tier.")

    log_gcp("WARNING", "All search tiers and links on AnimeSongLyrics failed verification.")
    return None, None

def scrape_translations_router(title: str, artists: str, baseline_df: pd.DataFrame, scraper_order: list = ["lyrical_nonsense", "lyrics_translate", "animeliryca", "anime_song_lyrics"]):
    baseline_text = "".join(baseline_df['jp_baseline'].tolist())

    for scraper in scraper_order:
        if scraper == "lyrical_nonsense":
            df, link = _scrape_utatime(title, artists, baseline_text)
        elif scraper == "lyrics_translate":
            df, link = _scrape_lyrics_translate(f"{title} {artists}", baseline_text)
        elif scraper == "animeliryca":
            df, link = _scrape_animeliryca(title, artists, baseline_text)
        elif scraper == "anime_song_lyrics":
            df, link = _scrape_anime_song_lyrics(title, artists, baseline_text)

        if df is not None:
            return df, link
    
    log_gcp("WARNING", "All translation sources failed to yield a matching result...", query=f"{title} {artists}")
    raise ScraperError("All translation sources failed to yield a matching result...", "EXPT_TRANS_NOT_FOUND")