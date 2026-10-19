#!/usr/bin/env python3
"""Render a thesis for visual review using PDFium and Pillow (no CJK CMap dependency)."""
import argparse
import json
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--detail', type=int, nargs='*', default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(args.pdf))
    inventory = []
    for start in range(0, len(doc), 20):
        sheet = Image.new('RGB', (1040, 1950), '#dddddd')
        draw = ImageDraw.Draw(sheet)
        for j in range(min(20, len(doc) - start)):
            page = doc[start + j]
            page.render(scale=.42).to_pil().save(args.output / f'page_{start+j+1:03}.png')
            im = Image.open(args.output / f'page_{start+j+1:03}.png').convert('RGB')
            sheet.paste(im, ((j % 4) * 260, (j // 4) * 390))
            draw.text(((j % 4) * 260 + 5, (j // 4) * 390 + 358), str(start + j + 1), fill='black')
            text_page = page.get_textpage()
            text = text_page.get_text_range()
            inventory.append({'page': start+j+1, 'characters': len(text.strip()), 'text': text})
            text_page.close()
            page.close()
        sheet.save(args.output / f'contact_{start+1:03}.jpg', quality=90)
    for number in args.detail:
        page = doc[number - 1]
        page.render(scale=1.8).to_pil().save(args.output / f'detail_{number:03}.png')
        page.close()
    (args.output / 'page_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2))
    print(f'Rendered {len(doc)} pages; detail pages: {args.detail}')
    doc.close()


if __name__ == '__main__':
    main()
