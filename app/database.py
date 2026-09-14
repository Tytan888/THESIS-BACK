import traceback
from google.cloud import firestore
from logger import log_gcp

project_id = "tyra-thesis-2"
db = firestore.Client(project=project_id)

def update_song_status_to_error(video_id: str, error_code: str = None):
    try:
        doc_data = {
            "status": "error"
        }
            
        if error_code is not None:
            doc_data["error_code"] = error_code

        doc_ref = db.collection("songs").document(video_id)
        doc_ref.set(doc_data, merge=True)

        # Delete the progress subcollection document to clean up
        db.collection("songs").document(video_id).collection("progress").document("status").delete()

        log_gcp("INFO", "Successfully updated song status to error.", video_id=video_id, status="error")
        
    except Exception as e:
        doc_ref = db.collection("songs").document(video_id)
        doc_ref.set({"status": "error", "error_code": "ERR_DB_ERROR_UPDATE_CRASH"}, merge=True)
        log_gcp("ERROR", "Failed to update database...", video_id=video_id, traceback=traceback.format_exc())

def save_pipeline_results(video_id: str, subtitles: list = None, vocab_map: list = None, translation_url: str = None):
    try:
        log_gcp("INFO", "Beginning global database save phase.", video_id=video_id)
        
        song_ref = db.collection("songs").document(video_id)
        batch = db.batch()
        op_count = 0

        if vocab_map:
            for vocab in vocab_map:
                if op_count >= 485: # Lowered slightly to leave room for the song update AND the progress deletion
                    batch.commit()
                    batch = db.batch()
                    op_count = 0
                
                uid = str(vocab["uid"])
                vocab_ref = db.collection("vocabulary").document(uid)
                batch.set(vocab_ref, {
                    "readings": vocab["readings"],
                    "writings": vocab["writings"],
                    "senses": vocab["senses"],
                    "appearing_in_songs": { video_id: vocab["first_seen_line"] }
                }, merge=True)
                
                op_count += 1

        song_data = {
            "status": "completed",
            "subtitles": subtitles if subtitles else [],
            "vocab_list": [str(v["uid"]) for v in vocab_map] if vocab_map else [],
            "translation_url": translation_url,
            "error_code": firestore.DELETE_FIELD,
        }
        
        batch.set(song_ref, song_data, merge=True)
        
        # Delete the progress subcollection document in the same batch
        progress_ref = song_ref.collection("progress").document("status")
        batch.delete(progress_ref)

        batch.commit()
        
        log_gcp("INFO", "Successfully finalized song and global vocabulary.", video_id=video_id)

    except Exception as e:
        doc_ref = db.collection("songs").document(video_id)
        doc_ref.set({"status": "error", "error_code": "ERR_DB_SAVING_CRASH"}, merge=True)
        log_gcp("ERROR", f"Failed to commit database updates: {e}", traceback=traceback.format_exc(), video_id=video_id)
        raise

@firestore.transactional
def _update_progress_txn(transaction, doc_ref, step_num, step_msg):
    snapshot = doc_ref.get(transaction=transaction)
    
    # Default to -1 if the document doesn't exist yet
    current_step = -1
    if snapshot.exists:
        current_step = snapshot.to_dict().get("processing_stage", -1)
        
    # Only update if the new step is further along in the pipeline
    if step_num > current_step:
        # Use set with merge=True so it creates the document if it doesn't exist
        transaction.set(doc_ref, {
            "processing_stage": step_num,
            "processing_msg": step_msg
        }, merge=True)

def update_processing_progress(video_id: str, step_num: int, step_msg: str):
    try:
        # Target the subcollection instead of the main document
        doc_ref = db.collection("songs").document(video_id).collection("progress").document("status")
        transaction = db.transaction()
        _update_progress_txn(transaction, doc_ref, step_num, step_msg)
    except Exception as e:
        log_gcp("WARNING", f"Failed to update progress to step {step_num}... {e}")