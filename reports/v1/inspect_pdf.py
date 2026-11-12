"""Optional PDF inspection (requires PyMuPDF and Pillow; not used for building)."""
from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / ".tools"))
import pymupdf
from PIL import Image, ImageDraw

doc = pymupdf.open(root / "report.pdf")
out = root / "build" / "preview"
out.mkdir(parents=True, exist_ok=True)
rows = []
tiles = []
for index, page in enumerate(doc):
    pix = page.get_pixmap(matrix=pymupdf.Matrix(.5, .5), alpha=False)
    tile = Image.new("RGB", (320, 455), "#e5e7eb")
    preview = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    preview.thumbnail((300, 425))
    tile.paste(preview, ((320-preview.width)//2, 25))
    ImageDraw.Draw(tile).text((10, 6), f"PDF page {index+1}", fill="black")
    tiles.append(tile)
    blocks = page.get_text("blocks")
    bad = [list(b[:4]) for b in blocks if b[0] < 0 or b[1] < 0 or b[2] > page.rect.width+1 or b[3] > page.rect.height+1]
    rows.append({"page": index+1, "text_characters": len(page.get_text()), "off_page_blocks": bad})
    if index in [0, 6, 8, 10, 11, 12, 13, 17, 20, 22, 24]:
        page.get_pixmap(matrix=pymupdf.Matrix(1.25,1.25), alpha=False).save(out / f"page-{index+1:02d}.png")
for start in range(0,len(tiles),12):
    part = tiles[start:start+12]
    sheet = Image.new("RGB",(320*4,455*((len(part)+3)//4)),"#e5e7eb")
    for i,tile in enumerate(part):sheet.paste(tile,((i%4)*320,(i//4)*455))
    sheet.save(out / f"contact-{start//12+1}.jpg")
(root / "build" / "report_text.txt").write_text("\n\f\n".join(p.get_text() for p in doc),encoding="utf-8")
result={"pages":len(doc),"pdf_sha256":hashlib.sha256((root/'report.pdf').read_bytes()).hexdigest(),"page_checks":rows}
(root / "build" / "pdf_inspection.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"pages":len(doc),"off_page_blocks":sum(len(r['off_page_blocks']) for r in rows),
                  "empty_pages":sum(r['text_characters']<=20 for r in rows)},indent=2))
assert not any(r['off_page_blocks'] for r in rows), 'Found content outside a PDF page'
assert all(r['text_characters']>20 for r in rows), 'Unexpected empty page'
