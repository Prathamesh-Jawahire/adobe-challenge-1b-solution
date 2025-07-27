import os
import json
import time
import datetime
import argparse
import pdfplumber
import numpy as np
from tqdm import tqdm
from collections import defaultdict
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import re
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline


# Load models
embedding_model = SentenceTransformer("/app/models/all-mpnet-base-v2" ,device='cpu')


# Works best till now, promising results.
tokenizer = AutoTokenizer.from_pretrained("/app/models/flan-t5-small-openai-feedback",local_files_only=True)
summarizer = AutoModelForSeq2SeqLM.from_pretrained("/app/models/flan-t5-small-openai-feedback",local_files_only=True)

def clean_summary(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    full_sentences = []
    word_count = 0
    for sentence in sentences:
        wc = len(sentence.split())
        if word_count + wc > 150:
            break
        full_sentences.append(sentence)
        word_count += wc
    return ' '.join(full_sentences)

def clean_text(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return ' '.join(lines)

def extract_text_from_page(pdf_path, page_num):
    with pdfplumber.open(pdf_path) as pdf:
        if 0 <= page_num < len(pdf.pages):
            text = pdf.pages[page_num].extract_text()
            return clean_text(text) if text else ''
        return ''

# def summarize_text(text, persona, job):
#     prompt = f"Summarize this content for a {persona} working on the task: '{job}'. Content: {text}"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
#     output_ids = summarizer.generate(input_ids, max_length=180, num_beams=4, early_stopping=True)
#     summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return clean_summary(summary)




def summarize_text(text, persona=None, job=None):
    # Ignore persona/job since this model isn't trained to use them
    prompt = f"Summarize: {text}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    output_ids = summarizer.generate(input_ids, max_length=180, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return clean_summary(summary)






# summarizer = pipeline("summarization", model="Falconsai/text_summarization", tokenizer="Falconsai/text_summarization")

# def summarize_text(text, persona=None, job=None):
#     # The model does not use persona/job — ignore them
#     result = summarizer(text, max_length=1000, min_length=30, do_sample=False)
#     return result[0]['summary_text']








def summarize_section(args):
    pdf_path, page_number, persona, job = args
    page_text = extract_text_from_page(pdf_path, page_number - 1)
    if not page_text:
        return None
    summary = summarize_text(page_text, persona, job)
    return {
        "document": os.path.basename(pdf_path),
        "page_number": page_number,
        "summary": summary
    }

def compute_similarity(text1, text2):
    emb1 = embedding_model.encode([text1])[0]
    emb2 = embedding_model.encode([text2])[0]
    return cosine_similarity([emb1], [emb2])[0][0]

def main(input_json_path, pdf_dir):
    start_time = time.time()

    with open(input_json_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    persona = input_data["persona"]["role"]
    job = input_data["job_to_be_done"]["task"]
    job_context = f"{persona}. {job}"
    document_list = input_data["documents"]

    document_scores = []
    timer_enforced = len(document_list) <= 5
    max_duration = 50 if timer_enforced else 90

    # Step 1: Identify relevant sections
    for doc in tqdm(document_list, desc="Scanning documents"):
        pdf_filename = doc["filename"]
        pdf_path = os.path.join(pdf_dir, pdf_filename)
        outline_path = os.path.join(pdf_dir, os.path.splitext(pdf_filename)[0] + ".json")

        if not os.path.exists(pdf_path) or not os.path.exists(outline_path):
            continue

        with open(outline_path, "r", encoding="utf-8") as f:
            outline_data = json.load(f)

        for entry in outline_data.get("outline", []):
            heading = entry.get("text", "").strip()
            page = entry.get("page")
            if not heading or page is None:
                continue

            sim = compute_similarity(heading, job_context)
            if sim < 0.25:
                continue

            document_scores.append({
                "document": pdf_filename,
                "page_number": page,
                "section_title": heading,
                "similarity": sim
            })

        if timer_enforced and time.time() - start_time > max_duration:
            break

    if not document_scores:
        output = {
            "metadata": {
                "input_documents": [doc["filename"] for doc in document_list],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            },
            "extracted_sections": [
                {
                    "document": None,
                    "section_title": "No relevant content found.",
                    "importance_rank": 1,
                    "page_number": None
                }
            ],
            "subsection_analysis": [
                {
                    "document": None,
                    "refined_text": "None of the documents provided contain relevant information for the persona and task described.",
                    "page_number": None
                }
            ]
        }
        with open("challenge1b_output.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
        return

    # Step 2: Select top 5 sections with doc diversity
    document_scores.sort(key=lambda x: -x["similarity"])

    final_sections = []
    summary_keys = set()  # Tracks (doc, page_number) we've already picked for summarization
    idx = 0

    while len(summary_keys) < 5 and idx < len(document_scores):
        score = document_scores[idx]
        key = (score["document"], score["page_number"])
        if key not in summary_keys:
            final_sections.append(score)
            summary_keys.add(key)
        idx += 1


    # Step 3: Summarize in parallel
    summary_inputs = [
        (
            os.path.join(pdf_dir, section["document"]),
            section["page_number"],
            persona,
            job
        )
        for section in final_sections
    ]

    # Deduplicate summary jobs based on (document, page_number)
    unique_summary_map = {}
    summary_args = []
    for section in final_sections:
        key = (section["document"], section["page_number"])
        if key not in unique_summary_map:
            unique_summary_map[key] = None  # placeholder
            summary_args.append((
                os.path.join(pdf_dir, section["document"]),
                section["page_number"],
                persona,
                job
            ))


    # Run summarization only once per unique (document, page_number)
    with ThreadPoolExecutor(max_workers=5) as executor:
        summary_results = list(executor.map(summarize_section, summary_args))

    # Assign summaries back to their keys
    for args, result in zip(summary_args, summary_results):
        doc_path, page_num, *_ = args
        doc_name = os.path.basename(doc_path)
        unique_summary_map[(doc_name, page_num)] = result

    # Rebuild outputs based on final_sections but now using mapped unique summaries
    extracted_sections = []
    subsection_analysis = []
    seen_summary_pages = set()
    valid_rank = 1

    for section in final_sections:
        key = (section["document"], section["page_number"])
        result = unique_summary_map.get(key)
        if result is None:
            continue

        extracted_sections.append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": valid_rank,
            "page_number": section["page_number"]
        })

        # Avoid duplicate summaries in subsection_analysis
        if key not in seen_summary_pages:
            subsection_analysis.append({
                "document": result["document"],
                "refined_text": result["summary"],
                "page_number": result["page_number"]
            })
            seen_summary_pages.add(key)

        valid_rank += 1




    output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in document_list],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    with open("challenge1b_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", required=True, help="Path to challenge1b_input.json")
#     parser.add_argument("--pdf_dir", required=True, help="Directory containing PDFs and outline JSONs")
#     args = parser.parse_args()
#     main(args.input, args.pdf_dir)


if __name__ == "__main__":
    input_json_path = "/app/challenge1b_input.json"
    pdf_dir = "/app/input"
    main(input_json_path, pdf_dir)