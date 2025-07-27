"""
PDF Outline Extractor 

Extracts document title and hierarchical headings from PDF files.
Features: Title extraction, table filtering, formatting validation, content analysis.
"""

import fitz
import json
import re
import unicodedata
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

PDF_DIR = "/app/input"
OUTPUT_DIR = "/app/input"


def normalize_text(text: str) -> str:
    """Clean and normalize text content."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def capitalize_first_letter(text: str) -> str:
    """Capitalize the first alphabetic character while preserving formatting."""
    if not text:
        return text
    
    for i, char in enumerate(text):
        if char.isalpha():
            return text[:i] + char.upper() + text[i+1:]
    
    return text


def is_signature_field(text: str) -> bool:
    """Check if text is a signature field."""
    text = normalize_text(text).lower().strip()
    signature_words = {
        'date', 'signature', 'sign', 'name', 'witness', 'seal', 'stamp',
        'approved', 'verified', 'checked', 'authorized', 'place'
    }
    clean_text = text.rstrip(':').strip()
    return clean_text in signature_words


def is_form_label(text: str) -> bool:
    """Check if text is a form label."""
    text = normalize_text(text).lower().strip()
    form_labels = {
        'address', 'phone', 'email', 'rsvp', 'contact', 'location', 'venue',
        'time', 'date', 'when', 'where', 'who', 'what', 'details', 'info',
        'information', 'notice', 'announcement', 'warning', 'attention',
        'note', 'memo', 'subject', 'from', 'to', 'cc', 'bcc'
    }
    clean_text = text.rstrip(':').strip()
    return clean_text in form_labels


def is_noise_content(text: str) -> bool:
    """Filter out noise content."""
    text = normalize_text(text)
    if not text or len(text) < 2:
        return True
    
    if re.fullmatch(r"\d+", text):
        return True
    
    date_patterns = [
        r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"[A-Za-z]{3,9}\s+\d{4}",
        r"\d{4}"
    ]
    for pattern in date_patterns:
        if re.fullmatch(pattern, text, re.IGNORECASE):
            return True
    
    if re.search(r'www\.|\.com|\.org|\.net|https?://', text.lower()):
        return True
    
    if re.fullmatch(r"\([^)]*\)", text):
        return True
    
    if re.fullmatch(r"[-_=*#~`^]{3,}", text):
        return True
    
    lower = text.lower()
    page_patterns = [
        r"page\s+\d+", r"figure\s+\d+", r"table\s+\d+", r"\d+\s*/\s*\d+"
    ]
    for pattern in page_patterns:
        if re.search(pattern, lower):
            return True
    
    if len(text.split()) > 15:
        return True
    
    if sum(c.isalnum() for c in text) / len(text) < 0.3:
        return True
    
    return False


def has_consistent_formatting(spans: List[Dict], font_tolerance: float = 1.5) -> bool:
    """Check if spans have consistent font size and boldness."""
    if not spans or len(spans) == 1:
        return True
    
    font_sizes = [span.get('size', 12) for span in spans if 'size' in span]
    if len(font_sizes) > 1:
        if max(font_sizes) - min(font_sizes) > font_tolerance:
            return False
    
    bold_flags = [bool(span.get('flags', 0) & 16) for span in spans]
    if len(set(bold_flags)) > 1:
        return False
    
    return True


def is_valid_heading(text: str) -> bool:
    """Validate if text qualifies as a heading."""
    text = normalize_text(text)
    
    if not any(c.isalpha() for c in text):
        return False
    
    if text.endswith(('.', ';', ',', '!', '?')):
        return False
    
    word_count = len(text.split())
    if word_count < 1 or word_count > 12:
        return False
    
    first_word = text.split()[0].lower() if text.split() else ""
    if first_word in ['the', 'a', 'an', 'this', 'that']:
        return False
    
    if is_signature_field(text):
        return False
    
    return True


def is_valid_title(text: str) -> bool:
    """Validate if text qualifies as a document title."""
    text = normalize_text(text)
    
    if not any(c.isalpha() for c in text):
        return False
    
    if is_form_label(text) or is_signature_field(text) or is_noise_content(text):
        return False
    
    if text.endswith(('.', ';', ',', '?')) and not text.endswith('!'):
        return False
    
    word_count = len(text.split())
    if word_count < 1 or word_count > 10:
        return False
    
    return True


def is_serial_number(text: str) -> bool:
    """Check if text represents a serial number or table numbering."""
    text = normalize_text(text).lower()
    
    serial_patterns = [
        r"^sr\.?\s*no\.?$", r"^s\.?\s*no\.?$", r"^sl\.?\s*no\.?$",
        r"^serial\s*no\.?$", r"^no\.?$", r"^#$", r"^item\s*no\.?$",
        r"^\d+\.?$", r"^\d+\)$", r"^[ivxlcdm]+\.?$", r"^[IVXLCDM]+\.?$"
    ]
    
    return any(re.match(pattern, text) for pattern in serial_patterns)


def detect_table_rows(all_lines: List[Dict]) -> List[float]:
    """Detect Y-coordinates of table rows based on serial numbers."""
    table_y_coordinates = set()
    
    for line in all_lines:
        if is_serial_number(line['text']):
            table_y_coordinates.add(line['bbox'][1])
    
    page_lines = {}
    for line in all_lines:
        page = line['page']
        if page not in page_lines:
            page_lines[page] = []
        page_lines[page].append(line)
    
    for page, lines in page_lines.items():
        x_groups = {}
        for line in lines:
            x = round(line['bbox'][0] / 10) * 10
            if x not in x_groups:
                x_groups[x] = []
            x_groups[x].append(line)
        
        for x, x_lines in x_groups.items():
            if len(x_lines) < 3:
                continue
            
            x_lines.sort(key=lambda l: l['bbox'][1])
            
            numbers = []
            y_coords = []
            for line in x_lines:
                if re.match(r'^\d+\.?$', line['text'].strip()):
                    try:
                        num = int(line['text'].strip().rstrip('.'))
                        numbers.append(num)
                        y_coords.append(line['bbox'][1])
                    except:
                        continue
            
            if len(numbers) >= 3:
                for i in range(len(numbers) - 2):
                    if (numbers[i+1] == numbers[i] + 1 and 
                        numbers[i+2] == numbers[i] + 2):
                        for j in range(i, min(i+3, len(y_coords))):
                            table_y_coordinates.add(y_coords[j])
                        break
    
    return list(table_y_coordinates)


def is_in_table_row(text_bbox: List[float], table_y_coords: List[float], tolerance: float = 8.0) -> bool:
    """Check if text is in a table row."""
    if not table_y_coords:
        return False
    
    text_y = text_bbox[1]
    return any(abs(text_y - table_y) <= tolerance for table_y in table_y_coords)


def filter_horizontal_table_headers(candidates: List[Dict], y_tolerance: float = 10.0, 
                                   min_horizontal_gap: float = 50.0) -> List[Dict]:
    """Remove horizontally aligned table headers."""
    if not candidates or len(candidates) < 2:
        return candidates
    
    candidates_sorted = sorted(candidates, key=lambda x: (x['page'], x['bbox'][1], x['bbox'][0]))
    
    filtered = []
    skip_indices = set()
    
    for i, current in enumerate(candidates_sorted):
        if i in skip_indices:
            continue
        
        current_y = current['bbox'][1]
        current_page = current['page']
        horizontal_neighbors = []
        
        for j in range(i + 1, len(candidates_sorted)):
            if j in skip_indices:
                continue
                
            other = candidates_sorted[j]
            
            if other['page'] != current_page:
                break
            
            other_y = other['bbox'][1]
            
            if abs(other_y - current_y) <= y_tolerance:
                current_x_end = current['bbox'][2]
                other_x_start = other['bbox'][0]
                horizontal_gap = other_x_start - current_x_end
                
                if horizontal_gap >= min_horizontal_gap:
                    horizontal_neighbors.append(j)
            elif other_y > current_y + y_tolerance:
                break
        
        if horizontal_neighbors:
            skip_indices.update(horizontal_neighbors)
        else:
            filtered.append(current)
    
    return filtered


def filter_inconsistent_formatting(candidates: List[Dict]) -> List[Dict]:
    """Filter out headings with inconsistent formatting."""
    filtered = []
    
    for candidate in candidates:
        spans = candidate.get('spans', [])
        if has_consistent_formatting(spans):
            filtered.append(candidate)
    
    return filtered


def is_paragraph_content(text: str) -> bool:
    """Check if text represents paragraph content."""
    text = normalize_text(text)
    
    if len(text.split()) < 3:
        return False
    
    if not any(c.islower() for c in text):
        return False
    
    if is_noise_content(text) or is_signature_field(text):
        return False
    
    if text.endswith(('.', '!', '?', ';')):
        return True
    
    lower_text = text.lower()
    paragraph_indicators = [
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'by', 'for', 
        'with', 'this', 'that', 'these', 'those'
    ]
    if any(word in lower_text.split() for word in paragraph_indicators):
        return True
    
    if (len(text.split()) >= 5 and 
        any(c.islower() for c in text) and 
        any(c.isupper() for c in text)):
        return True
    
    return False


def has_paragraph_below(heading_idx: int, all_lines: List[Dict], 
                       heading_candidates: List[Dict], max_distance: int = 15) -> bool:
    """Check if heading has paragraph content below it."""
    if heading_idx >= len(all_lines):
        return False
    
    heading = all_lines[heading_idx]
    candidate_texts = {candidate['text'] for candidate in heading_candidates}
    
    lines_checked = 0
    
    for i in range(heading_idx + 1, len(all_lines)):
        if lines_checked >= max_distance:
            break
            
        next_line = all_lines[i]
        lines_checked += 1
        
        if next_line['page'] > heading['page'] + 2:
            break
        
        if next_line['text'] in candidate_texts:
            continue
        
        if is_paragraph_content(next_line['text']):
            return True
    
    return False


def filter_headings_with_content(candidates: List[Dict], all_lines: List[Dict]) -> List[Dict]:
    """Filter headings that don't have content below them."""
    text_to_index = {}
    for i, line in enumerate(all_lines):
        text_to_index[line['text']] = i
    
    filtered_candidates = []
    
    for candidate in candidates:
        heading_text = candidate['text']
        
        if heading_text in text_to_index:
            heading_idx = text_to_index[heading_text]
            
            if has_paragraph_below(heading_idx, all_lines, candidates):
                filtered_candidates.append(candidate)
        else:
            filtered_candidates.append(candidate)
    
    return filtered_candidates


def extract_title(first_page_elements: List[Dict], median_size: float, 
                 page_height: float = 792) -> str:
    """Extract title from first 60% of first page."""
    if not first_page_elements:
        return ""
    
    sixty_percent_height = page_height * 0.6
    top_section_elements = [
        elem for elem in first_page_elements 
        if elem['bbox'][1] <= sixty_percent_height
    ]
    
    if not top_section_elements:
        return ""
    
    title_candidates = []
    for element in top_section_elements:
        text = normalize_text(element['text'])
        
        if not is_valid_title(text):
            continue
        
        if text.startswith('(') and text.endswith(')'):
            continue
        
        title_candidates.append(element)
    
    if not title_candidates:
        return ""
    
    largest_size = max(elem['size'] for elem in title_candidates)
    largest_candidates = [elem for elem in title_candidates 
                         if elem['size'] >= largest_size - 1.0]
    
    if len(largest_candidates) == 1:
        return capitalize_first_letter(largest_candidates[0]['text'])
    
    scored_candidates = []
    for element in largest_candidates:
        text = normalize_text(element['text'])
        score = 0
        
        size_ratio = element['size'] / median_size
        if size_ratio > 2.0:
            score += 10
        elif size_ratio > 1.5:
            score += 8
        elif size_ratio > 1.2:
            score += 6
        
        y_pos = element['bbox'][1]
        position_ratio = y_pos / sixty_percent_height
        if position_ratio < 0.3:
            score += 4
        elif position_ratio < 0.6:
            score += 2
        
        if element['is_bold']:
            score += 3
        
        word_count = len(text.split())
        if 1 <= word_count <= 3:
            score += 4
        elif 4 <= word_count <= 6:
            score += 2
        elif word_count > 10:
            score -= 2
        
        if text.isupper():
            score += 2
        
        page_width = 612
        x_center = element['bbox'][0] + (element['bbox'][2] - element['bbox'][0]) / 2
        if abs(x_center - page_width/2) < 100:
            score += 2
        
        scored_candidates.append((score, text, element['size']))
    
    if scored_candidates:
        scored_candidates.sort(reverse=True)
        return capitalize_first_letter(scored_candidates[0][1])
    
    return ""


def extract_pdf_outline(pdf_path: str) -> Dict[str, Any]:
    """Extract outline from PDF with title and hierarchical headings."""
    try:
        doc = fitz.open(pdf_path)
        
        all_lines = []
        first_page_elements = []
        font_sizes = []
        bold_count = 0
        page_height = 792
        
        for page_num, page in enumerate(doc):
            try:
                if page_num == 0:
                    page_height = page.rect.height
                
                blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                        
                    for line in block["lines"]:
                        if "spans" not in line or not line["spans"]:
                            continue
                        
                        spans = line["spans"]
                        text = "".join(span.get("text", "") for span in spans).strip()
                        
                        if not text or is_noise_content(text):
                            continue
                        
                        sizes = [span.get("size", 12) for span in spans if "size" in span]
                        if not sizes:
                            continue
                            
                        font_size = max(sizes)
                        is_bold = any(span.get("flags", 0) & 16 for span in spans)
                        
                        font_sizes.append(font_size)
                        if is_bold:
                            bold_count += 1
                        
                        line_data = {
                            "text": normalize_text(text),
                            "size": font_size,
                            "is_bold": is_bold,
                            "bbox": line.get("bbox", [0, 0, 0, 0]),
                            "page": page_num,
                            "spans": spans
                        }
                        
                        all_lines.append(line_data)
                        
                        if page_num == 0:
                            first_page_elements.append(line_data)
                            
            except Exception:
                continue
        
        doc.close()
        
        if not all_lines:
            return {"title": "", "outline": []}
        
        title = extract_title(first_page_elements, 
                            np.median(font_sizes) if font_sizes else 12, 
                            page_height)
        
        table_y_coords = detect_table_rows(all_lines)
        
        if not font_sizes:
            return {"title": title, "outline": []}
            
        median_size = np.median(font_sizes)
        bold_frequency = bold_count / len(all_lines) if all_lines else 0
        
        candidates = []
        
        for i, line in enumerate(all_lines):
            if (title and 
                normalize_text(line['text']).lower() == normalize_text(title).lower()):
                continue
            
            if not is_valid_heading(line['text']):
                continue
            
            if is_in_table_row(line['bbox'], table_y_coords):
                continue
            
            size_ratio = line['size'] / median_size
            
            vertical_gap = 0
            if i > 0 and all_lines[i-1]['page'] == line['page']:
                vertical_gap = line['bbox'][1] - all_lines[i-1]['bbox'][3]
            
            score = 0
            
            if size_ratio > 1.3:
                score += 4
            elif size_ratio > 1.15 and line['is_bold']:
                score += 3
            elif line['is_bold'] and bold_frequency < 0.2:
                score += 3
            
            if vertical_gap > 20:
                score += 2
            
            if re.match(r'^\d+[\.\)]\s', line['text']):
                score += 3
            
            words = line['text'].split()
            if len(words) <= 6 and line['is_bold']:
                score += 2
            
            title_ratio = (sum(1 for w in words if w and w[0].isupper()) / 
                          max(1, len(words)))
            if title_ratio > 0.7:
                score += 2
            
            if score >= 4:
                line['score'] = score
                line['vertical_gap'] = vertical_gap
                line['size_ratio'] = size_ratio
                candidates.append(line)
        
        if candidates:
            candidates = filter_inconsistent_formatting(candidates)
            candidates = filter_horizontal_table_headers(candidates)
            candidates = filter_headings_with_content(candidates, all_lines)
        
        if not candidates:
            return {"title": title, "outline": []}
        
        if len(candidates) >= 2:
            features = []
            for c in candidates:
                features.append([
                    c['size'], int(c['is_bold']), c['bbox'][0],
                    c['vertical_gap'], len(c['text']), c['score']
                ])
            
            X = np.array(features)
            X = StandardScaler().fit_transform(X)
            
            n_clusters = min(3, max(2, len(candidates) // 2))
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels = clustering.fit_predict(X)
            
            cluster_sizes = {}
            for i, label in enumerate(labels):
                if label not in cluster_sizes:
                    cluster_sizes[label] = []
                cluster_sizes[label].append(candidates[i]['size'])
            
            sorted_clusters = sorted(cluster_sizes.keys(), 
                                   key=lambda x: np.mean(cluster_sizes[x]), 
                                   reverse=True)
            
            level_map = {cluster: f"H{min(i+1, 3)}" 
                        for i, cluster in enumerate(sorted_clusters)}
            
            for i, candidate in enumerate(candidates):
                candidate['level'] = level_map[labels[i]]
        else:
            candidates[0]['level'] = 'H1'
        
        outline = []
        for candidate in candidates:
            capitalized_text = capitalize_first_letter(candidate['text'])
            outline.append({
                "level": candidate.get('level', 'H1'),
                "text": capitalized_text,
                "page": candidate['page']
            })
        
        outline.sort(key=lambda x: (
            x['page'], 
            next(c['bbox'][1] for c in candidates 
                 if normalize_text(c['text']).lower() == normalize_text(x['text']).lower())
        ))
        
        seen = set()
        clean_outline = []
        for item in outline:
            normalized = item['text'].lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                clean_outline.append(item)
        
        return {"title": title, "outline": clean_outline}
        
    except Exception:
        return {"title": "", "outline": []}


def process_all():
    """Process all PDFs in the input directory."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    for pdf_file in Path(PDF_DIR).glob("*.pdf"):
        try:
            result = extract_pdf_outline(str(pdf_file))
            output_path = Path(OUTPUT_DIR) / f"{pdf_file.stem}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
        except Exception:
            continue


if __name__ == "__main__":
    process_all()
