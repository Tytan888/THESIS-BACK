from fastapi import FastAPI, Request
import traceback

import scraper
import cleaning
import alignment
import database
from logger import log_gcp, ProcessingStep

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "Eventarc listener is active."}

@app.post("/")
async def eventarc_handler(request: Request):
    subject = request.headers.get("ce-subject")
    if not subject:
        log_gcp("WARNING", "Received CloudEvent without 'ce-subject' header. Ignoring event...", headers=dict(request.headers))
        return {"status": "ignored", "reason": "Not a valid CloudEvent..."}
    
    parts = subject.split("/")
    if len(parts) < 2 or parts[-2] != "songs":
        log_gcp("ERROR", "Received CloudEvent with invalid subject. Ignoring event...", headers=dict(request.headers))
        return {"status": "ignored", "reason": "Not a songs collection event..."}

    video_id = parts[-1]
    doc_ref = database.db.collection("songs").document(video_id)
    doc = doc_ref.get()

    if not doc.exists:
        log_gcp("ERROR", f"Document for video_id {video_id} does not exist. It may have been deleted. Ignoring event...", video_id=video_id)
        return {"status": "ignored", "reason": "Document was deleted..."}

    doc_data = doc.to_dict()
    if doc_data.get("status") != "processing":
        log_gcp("DEBUG", f"Ignoring event for {video_id}. Status is '{doc_data.get('status')}'.")
        return {"status": "ignored", "reason": "Status is not processing."}

    log_gcp("INFO", f"Eventarc triggered processing for {video_id}")
    generate_subs_pipeline(video_id)
    
    return {"status": "success", "video_id": video_id}

def generate_subs_pipeline(video_id: str):

    def update_progress(step: ProcessingStep):
        database.update_processing_progress(video_id, step.value, step.msg)

    def check_for_ghost_instance(video_id: str) -> bool:
        current_doc = database.db.collection("songs").document(video_id).get()
        if current_doc.exists and current_doc.to_dict().get("status") == "completed":
            log_gcp("WARNING", f"Ghost instance detected! Song {video_id} was already completed by another worker. Aborting save.")
            return True
        return False

    try:
        update_progress(ProcessingStep.JUST_RECEIVED)

        # STEP 1: Scrape
        update_progress(ProcessingStep.SCRAPING_BASELINE)
        baseline_df, title, artists = scraper.scrape_baseline_youtube(video_id)
        update_progress(ProcessingStep.SCRAPING_TRANSLATION)
        translation_df, translation_url = scraper.scrape_translations_router(title=title, artists=artists, baseline_df=baseline_df)
        # STEP 1.5: Check for ghost instance (another worker may have completed this song while we were processing)
        if check_for_ghost_instance(video_id=video_id): return

        # STEP 2: Align & Clean
        clean_df = cleaning.run_cleaning_pipeline(baseline_df, translation_df, progress_cb=update_progress)
        # STEP 2.5: Check for ghost instance (another worker may have completed this song while we were processing)
        if check_for_ghost_instance(video_id=video_id): return

        # STEP 3: ML Word Alignments
        final_df, vocab_map = alignment.run_alignment_pipeline(clean_df, progress_cb=update_progress)
        subtitles_list = final_df.fillna("").to_dict(orient="records")
        # STEP 3.5: Check for ghost instance (another worker may have completed this song while we were processing)
        if check_for_ghost_instance(video_id=video_id): return

        # STEP 4: Save & Upload
        update_progress(ProcessingStep.SAVING_DB)
        database.save_pipeline_results(video_id, subtitles_list, vocab_map, translation_url)

    except (scraper.ScraperError, cleaning.CleaningError, alignment.AlignmentError) as e:
        current_doc = database.db.collection("songs").document(video_id).get()
        if current_doc.exists and current_doc.to_dict().get("status") == "completed":
            log_gcp("WARNING", f"Pipeline crashed, but another instance already succeeded. Aborting error log.")
            return
        
        database.update_song_status_to_error(
            video_id=video_id, 
            error_code=e.error_code
        )

    except Exception as e:
        log_gcp("ERROR", f"Pipeline failed for {video_id}... {e}", traceback=traceback.format_exc(), video_id=video_id)
        
        current_doc = database.db.collection("songs").document(video_id).get()
        if current_doc.exists and current_doc.to_dict().get("status") == "completed":
            log_gcp("WARNING", f"Pipeline crashed, but another instance already succeeded. Aborting error log.")
            return

        database.update_song_status_to_error(
            video_id=video_id, 
            error_code="ERR_UNKNOWN_CRASH"
        )