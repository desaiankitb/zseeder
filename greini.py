import argparse
import csv
import json
import mimetypes
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from google import genai
from google.genai import types

from response_model import PrescriptionBatch


GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
DEFAULT_DATASET_DIR = Path("zseeder/data/validation_dataset/test")
TEXT_FIELDS = (
    "doctor_name",
    "clinic_name",
    "clinic_address",
    "patient_name",
    "date",
    "signature",
)

DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%B %d %Y",
    "%B %d, %Y",
    "%b %d %Y",
    "%b %d, %Y",
)

_CLIENT: Optional[genai.Client] = None


def get_genai_client() -> genai.Client:
    """
    Lazily instantiate a single Gemini client so benchmarking runs do not
    recreate HTTP sessions for every image.
    """
    global _CLIENT
    if _CLIENT is None:
        api_key = "AIzaSyD2z4azoLRBHvmgfWaP6Q-tBJ6WJjRv60cabd"
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def _load_image(image_path: Path) -> Tuple[bytes, str]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    return image_path.read_bytes(), mime_type


def perform_ocr_with_gemini(image_path: Path) -> PrescriptionBatch:
    """
    Performs OCR on an image using Google's official GenAI client and returns structured data.
    """
    image_bytes, mime_type = _load_image(image_path)
    client = get_genai_client()

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part.from_text(
                    text=(
                        "You are an expert pharmacy technician. Extract every prescription "
                        "from this document and return JSON that matches the provided schema."
                    )
                ),
            ],
        )
    ]

    config: Dict[str, Any] = {
        "response_mime_type": "application/json",
        "response_json_schema": PrescriptionBatch.model_json_schema(),
    }

    try:
        config["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)
    except TypeError:
        pass

    stream = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=contents,
        config=config,
    )

    chunks: List[str] = []
    for chunk in stream:
        text = getattr(chunk, "text", None)
        if not text and getattr(chunk, "candidates", None):
            try:
                text = chunk.candidates[0].content.parts[0].text
            except (AttributeError, IndexError):
                text = None

        if text:
            chunks.append(text)

    if not chunks:
        raise RuntimeError("No text returned from Gemini.")

    raw_json = "".join(chunks).strip()
    batch = PrescriptionBatch.model_validate_json(raw_json)
    batch.document_name = batch.document_name or image_path.name
    return batch


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip().lower()


def normalize_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    cleaned = value.strip()
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", " ").replace(".", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    digits_only = re.sub(r"[^\d]", "", cleaned)
    if len(digits_only) == 8:
        try:
            dt = datetime.strptime(digits_only, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return normalize_text(value)


def normalize_field_value(field_name: str, value: Optional[str]) -> str:
    if not value:
        return ""
    if field_name == "date":
        normalized = normalize_date(value)
        return normalized or ""
    return normalize_text(value)


def parse_medications_block(value: str) -> List[Dict[str, str]]:
    """
    Parses the flat medication list stored in the annotations into a list of
    descriptor/instruction pairs.
    """
    if not value:
        return []

    tokens = [token.strip() for token in value.split("-") if token.strip()]
    meds: List[Dict[str, str]] = []
    for idx in range(0, len(tokens), 2):
        descriptor = tokens[idx]
        instructions = tokens[idx + 1] if idx + 1 < len(tokens) else ""
        meds.append({"descriptor": descriptor, "instructions": instructions})
    return meds


def parse_ground_truth(annotation_path: Path) -> Dict[str, Any]:
    data = json.loads(annotation_path.read_text())
    raw = data.get("ground_truth", "")
    cleaned = raw.replace("<s_ocr>", "").replace("</s>", "").strip()
    matches = list(re.finditer(r"([a-z_]+):", cleaned))

    parsed: Dict[str, Any] = {}
    for idx, match in enumerate(matches):
        key = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        value = cleaned[start:end].strip()
        parsed[key] = value

    if "medications" in parsed:
        parsed["medications"] = parse_medications_block(parsed["medications"])
    return parsed


def sanitize_medication_entries(entries: Iterable[Dict[str, str]]) -> List[str]:
    serialised: List[str] = []
    for entry in entries:
        descriptor = (entry.get("descriptor") or entry.get("name") or "").strip()
        strength = entry.get("strength", "").strip()
        directions = (
            entry.get("instructions")
            or entry.get("instruction")
            or entry.get("directions")
            or ""
        ).strip()

        medication_text = " ".join(filter(None, [descriptor, strength])).strip()
        if directions:
            medication_text = f"{medication_text} | {directions}"

        if medication_text:
            serialised.append(medication_text)
    return serialised


def prediction_to_fields(batch: PrescriptionBatch) -> Dict[str, Any]:
    if not batch.prescriptions:
        return {}

    prescription = batch.prescriptions[0]
    meds = []
    for detail in prescription.prescription:
        meds.append(
            {
                "descriptor": detail.drug_name or "",
                "strength": detail.strength or "",
                "directions": detail.directions or "",
            }
        )

    first_detail = prescription.prescription[0] if prescription.prescription else None
    return {
        "doctor_name": prescription.prescriber.name,
        "clinic_name": prescription.prescriber.practice_name
        or prescription.prescriber.name,
        "clinic_address": prescription.prescriber.address,
        "patient_name": prescription.patient.name,
        "date": first_detail.date_of_issue if first_detail else None,
        "signature": prescription.prescriber.name,
        "medications": sanitize_medication_entries(meds),
    }


def evaluate_text_field(
    field_name: str, ground_truth: Optional[str], prediction: Optional[str]
) -> Dict[str, Any]:
    gt_present = bool(ground_truth and ground_truth.strip())
    pred_present = bool(prediction and prediction.strip())
    normalized_gt = normalize_field_value(field_name, ground_truth)
    normalized_pred = normalize_field_value(field_name, prediction)
    match = gt_present and pred_present and normalized_gt == normalized_pred
    return {
        "gt": ground_truth,
        "pred": prediction,
        "gt_present": gt_present,
        "pred_present": pred_present,
        "normalized_gt": normalized_gt,
        "normalized_pred": normalized_pred,
        "match": match,
    }


def evaluate_medications_field(
    ground_truth: Optional[List[Dict[str, str]]], prediction: Optional[List[str]]
) -> Dict[str, Any]:
    gt_serial = sanitize_medication_entries(ground_truth or [])
    pred_serial = prediction or []

    gt_norm = [normalize_text(item) for item in gt_serial if normalize_text(item)]
    pred_norm = [normalize_text(item) for item in pred_serial if normalize_text(item)]

    gt_set = set(gt_norm)
    pred_set = set(pred_norm)
    true_positive = len(gt_set & pred_set)
    precision = true_positive / len(pred_set) if pred_set else 0.0
    recall = true_positive / len(gt_set) if gt_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "gt": gt_serial,
        "pred": pred_serial,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": gt_norm == pred_norm and bool(gt_norm),
    }


def evaluate_sample(gt_fields: Dict[str, Any], pred_fields: Dict[str, Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for field in TEXT_FIELDS:
        results[field] = evaluate_text_field(field, gt_fields.get(field), pred_fields.get(field))
    results["medications"] = evaluate_medications_field(
        gt_fields.get("medications"), pred_fields.get("medications")
    )
    return results


def aggregate_text_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = sum(1 for record in records if record["gt_present"])
    if total == 0:
        return {"samples": 0, "accuracy": None, "coverage": None}

    matches = sum(1 for record in records if record["gt_present"] and record["match"])
    covered = sum(
        1 for record in records if record["gt_present"] and record["pred_present"]
    )
    return {
        "samples": total,
        "accuracy": matches / total,
        "coverage": covered / total,
    }


def aggregate_medication_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"samples": 0, "precision": None, "recall": None, "f1": None, "exact": None}

    precisions = [record["precision"] for record in records]
    recalls = [record["recall"] for record in records]
    f1s = [record["f1"] for record in records]
    exact = [1.0 if record["exact_match"] else 0.0 for record in records]

    return {
        "samples": len(records),
        "precision": mean(precisions),
        "recall": mean(recalls),
        "f1": mean(f1s),
        "exact": mean(exact),
    }

def write_metrics_csv(
    output_path: Path,
    text_metrics: Dict[str, Dict[str, Any]],
    medication_metrics: Dict[str, Any],
) -> None:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "metric",
        "samples",
        "accuracy",
        "coverage",
        "precision",
        "recall",
        "f1",
        "exact_match",
    ]

    def fmt(value: Optional[float]) -> str:
        return "" if value is None else f"{value:.4f}"

    rows: List[Dict[str, Any]] = []
    for field, metrics in text_metrics.items():
        rows.append(
            {
                "metric": field,
                "samples": metrics["samples"],
                "accuracy": fmt(metrics.get("accuracy")),
                "coverage": fmt(metrics.get("coverage")),
                "precision": "",
                "recall": "",
                "f1": "",
                "exact_match": "",
            }
        )

    rows.append(
        {
            "metric": "medications",
            "samples": medication_metrics["samples"],
            "accuracy": "",
            "coverage": "",
            "precision": fmt(medication_metrics.get("precision")),
            "recall": fmt(medication_metrics.get("recall")),
            "f1": fmt(medication_metrics.get("f1")),
            "exact_match": fmt(medication_metrics.get("exact")),
        }
    )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_details_csv(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = ["sample"]
    for field in TEXT_FIELDS:
        fieldnames.extend(
            [
                f"{field}_gt",
                f"{field}_pred",
                f"{field}_norm_gt",
                f"{field}_norm_pred",
                f"{field}_match",
            ]
        )
    fieldnames.extend(
        [
            "medications_gt",
            "medications_pred",
            "medications_precision",
            "medications_recall",
            "medications_f1",
            "medications_exact_match",
        ]
    )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    dataset_dir: Path,
    sample_limit: Optional[int] = None,
    summary_csv: Optional[Path] = None,
    details_csv: Optional[Path] = None,
) -> None:
    annotations_dir = dataset_dir / "annotations"
    images_dir = dataset_dir / "images"
    annotation_paths = sorted(annotations_dir.glob("*.json"))

    text_field_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    medication_records: List[Dict[str, Any]] = []
    sample_details: List[Dict[str, Any]] = []

    processed = 0
    for annotation_path in annotation_paths:
        if sample_limit is not None and processed >= sample_limit:
            break

        image_path = images_dir / (annotation_path.stem + ".png")
        if not image_path.exists():
            print(f"Skipping {annotation_path.stem}: missing image {image_path}")
            continue

        try:
            gt_fields = parse_ground_truth(annotation_path)
            prediction = perform_ocr_with_gemini(image_path)
            pred_fields = prediction_to_fields(prediction)
            evaluation = evaluate_sample(gt_fields, pred_fields)
        except Exception as exception:  # pragma: no cover - diagnostic output
            print(f"[ERROR] {annotation_path.stem}: {exception}")
            continue

        for field in TEXT_FIELDS:
            text_field_records[field].append(evaluation[field])
        medication_records.append(evaluation["medications"])

        detail_row: Dict[str, Any] = {"sample": annotation_path.stem}
        for field in TEXT_FIELDS:
            field_eval = evaluation[field]
            detail_row[f"{field}_gt"] = field_eval.get("gt") or ""
            detail_row[f"{field}_pred"] = field_eval.get("pred") or ""
            detail_row[f"{field}_norm_gt"] = field_eval.get("normalized_gt") or ""
            detail_row[f"{field}_norm_pred"] = field_eval.get("normalized_pred") or ""
            detail_row[f"{field}_match"] = field_eval.get("match")

        meds_eval = evaluation["medications"]
        detail_row["medications_gt"] = " || ".join(meds_eval.get("gt") or [])
        detail_row["medications_pred"] = " || ".join(meds_eval.get("pred") or [])
        detail_row["medications_precision"] = meds_eval.get("precision")
        detail_row["medications_recall"] = meds_eval.get("recall")
        detail_row["medications_f1"] = meds_eval.get("f1")
        detail_row["medications_exact_match"] = meds_eval.get("exact_match")

        sample_details.append(detail_row)
        processed += 1

    if processed == 0:
        print("No samples were processed; ensure the dataset path is correct.")
        return

    print(f"\nProcessed {processed} samples from {dataset_dir}\n")
    print("Field-level accuracy (exact string match, lowercased/normalized):")
    text_metrics_summary: Dict[str, Dict[str, Any]] = {}
    for field in TEXT_FIELDS:
        metrics = aggregate_text_metrics(text_field_records[field])
        text_metrics_summary[field] = metrics
        accuracy = metrics["accuracy"]
        coverage = metrics["coverage"]
        samples = metrics["samples"]
        accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "n/a"
        coverage_text = f"{coverage:.2%}" if coverage is not None else "n/a"
        print(
            f"- {field:15s} | samples={samples:2d} | accuracy={accuracy_text} | coverage={coverage_text}"
        )

    med_metrics = aggregate_medication_metrics(medication_records)
    print("\nMedication list metrics (set overlap on normalized entries):")
    print(
        f"- precision={med_metrics['precision']:.2%} "
        f"recall={med_metrics['recall']:.2%} "
        f"f1={med_metrics['f1']:.2%} "
        f"exact_list_match={med_metrics['exact']:.2%}"
    )

    summary_path = summary_csv or (dataset_dir / "benchmark_metrics.csv")
    write_metrics_csv(summary_path, text_metrics_summary, med_metrics)
    print(f"\nSaved metrics CSV to {summary_path}")

    details_path = details_csv or (dataset_dir / "benchmark_details.csv")
    write_details_csv(details_path, sample_details)
    print(f"Saved prediction details CSV to {details_path}")


def ensure_demo_image(image_path: Path) -> None:
    if image_path.exists():
        return

    if image_path.name != "sample_invoice.png":
        raise FileNotFoundError(f"Image '{image_path}' not found.")

    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (600, 400), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        draw.text((50, 50), "INVOICE", fill=(0, 0, 0), font=font)
        draw.text((50, 100), "Billed to: John Doe", fill=(0, 0, 0), font=font)
        draw.text((50, 150), "Item: 1x Widget - $50.00", fill=(0, 0, 0), font=font)
        draw.text((50, 200), "Item: 2x Gadget - $75.00", fill=(0, 0, 0), font=font)
        draw.text((50, 250), "Total: $125.00", fill=(0, 0, 0), font=font)

        img.save(image_path)
    except ImportError:
        raise FileNotFoundError(
            "Pillow is not installed, and the sample image could not be created."
        )


def run_single_image_flow(image_path: Path) -> None:
    ensure_demo_image(image_path)
    print(f"Performing OCR on '{image_path}' using Gemini...")
    result = perform_ocr_with_gemini(image_path)
    print("\n--- Structured Prescription ---")
    print(result.model_dump_json(indent=2))
    print("-------------------------------")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gemini-powered OCR on a single prescription image or benchmark "
            "against a labeled dataset."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Run OCR on a single image instead of the full dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset directory containing 'images/' and 'annotations/' subfolders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of samples to benchmark (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path to save aggregated metrics as CSV (defaults to dataset_dir/benchmark_metrics.csv).",
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        default=None,
        help="Optional path to save ground-truth vs prediction rows (defaults to dataset_dir/benchmark_details.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image:
        run_single_image_flow(args.image)
    else:
        run_benchmark(
            args.dataset,
            sample_limit=args.limit,
            summary_csv=args.summary_csv,
            details_csv=args.details_csv,
        )


if __name__ == "__main__":
    main()
