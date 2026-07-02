# Newspaper Layout Detection & Classification

A computer vision pipeline that takes a newspaper image and automatically detects every region on the page — articles, headlines, and ads — then reads the article text and figures out what topic each one is about.

Built with Faster R-CNN for detection, EasyOCR for text extraction, and DistilBERT for classification. Served as a live REST API via FastAPI.

---

## What it does

Upload a newspaper scan and get back:
- Bounding boxes around every article, headline, and ad
- The extracted text from each article region
- A topic category for each article (Politics, Sports, Crime, etc.)
- An annotated image with color-coded boxes you can download

```
Input: newspaper image (JPG/PNG)
       ↓
Detectron2  →  finds regions (article / headline / ad)
       ↓
EasyOCR     →  reads text from each article crop
       ↓
DistilBERT  →  classifies each article into 1 of 15 categories
       ↓
Output: JSON response  +  annotated PNG
```

---

## API Endpoints

| Method | Endpoint | Returns |
|--------|----------|---------|
| GET | `/health` | Server status |
| POST | `/analyze` | JSON with all detected regions + categories |
| POST | `/analyze/visualize` | Annotated PNG with colored bounding boxes |

### Sample `/analyze` response

```json
{
  "summary": {
    "total_regions": 10,
    "articles": 6,
    "headlines": 1,
    "ads": 3,
    "categories_found": ["Politics", "International", "Crime & Law"]
  },
  "regions": [
    {
      "box": [37.4, 560.3, 241.4, 1397.1],
      "type": "article",
      "score": 0.993,
      "text": "CPI(M) leader Mythily Sivaraman passes away...",
      "category": "Politics",
      "confidence": 0.877
    }
  ]
}
```

### Visualization output

The `/analyze/visualize` endpoint returns a PNG where:
- **Article boxes** are color-coded by topic category
- **Headline boxes** are shown in green
- **Ad boxes** are shown in red

Each box is labeled with the category name and confidence score.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Layout detection | Detectron2 — Faster R-CNN with ResNet-50 FPN backbone |
| Text extraction | EasyOCR |
| Article classification | DistilBERT (fine-tuned on HuffPost news dataset) |
| API server | FastAPI + Uvicorn |
| Public tunnel | ngrok |
| Training environment | Google Colab (T4 GPU) |

---

## Models

### Layout Detection — Faster R-CNN
- **Backbone:** ResNet-50 with Feature Pyramid Network
- **Pretrained on:** COCO — fine-tuned on custom newspaper dataset
- **Classes:** `article`, `headline`, `ads`
- **Training:** 3000 iterations, batch size 2, LR 0.00025
- **Inference thresholds:** confidence > 0.7, NMS IoU < 0.3

### Article Classification — DistilBERT
- **Base model:** `distilbert-base-uncased`
- **Dataset:** HuffPost News Category Dataset (~200k articles)
- **Categories:** 15 (Politics, Sports, Crime & Law, Entertainment, etc.)
- **Training:** 3 epochs, LR 2e-5, batch size 32
- **Input:** headline + article description concatenated

---

## Project Structure

```
Newspaper_Project/
│
├── Detectron2.ipynb          # Data prep, training, evaluation
├── BERT.ipynb                # DistilBERT fine-tuning
├── OCR.ipynb                 # Full pipeline demo + visualization
├── fastAPI.ipynb             # REST API server
│
├── model_output/
│   ├── model_final.pth       # Trained Faster R-CNN weights
│   └── config.yaml           # Detectron2 config
│
├── news_classifier/
│   └── final_model/          # Saved DistilBERT + tokenizer
│
└── coco_dataset/
    └── annotations/
        ├── train.json
        └── val.json
```

---

## Running the API

1. Open `fastAPI.ipynb` in Google Colab
2. Mount your Google Drive (models are loaded from there)
3. Run all cells top to bottom
4. Copy the ngrok public URL from the output
5. Hit `/docs` on that URL to open the interactive Swagger UI

```bash
# Test with curl
curl -X POST "https://your-ngrok-url/analyze" \
  -H "accept: application/json" \
  -F "file=@newspaper.jpg"

# Get the annotated image
curl -X POST "https://your-ngrok-url/analyze/visualize" \
  -F "file=@newspaper.jpg" \
  --output result.png
```

---

## 15 Article Categories

Politics · Business & Economy · Sports · Entertainment · Science & Technology · Health & Medicine · Crime & Law · Education · Environment · Lifestyle · Travel · Religion & Culture · Parenting · Opinion & Editorial · International

---

