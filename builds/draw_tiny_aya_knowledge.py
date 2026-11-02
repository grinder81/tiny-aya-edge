"""Draw source-derived architecture schematics without loading a model.

SVG needs only the standard library. --png additionally uses Pillow for previews.
"""
import argparse
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/knowledge-bank/diagrams'
INK = '#17324d'
MUTED = '#526678'
BLUE = '#e9f2ff'
GREEN = '#e6f5ee'
PURPLE = '#eeeaff'
ORANGE = '#fff0d9'


class Canvas:
    def __init__(self, height, title, subtitle, png=False):
        self.width, self.height = 1000, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" role="img" aria-label="{html.escape(title)}">',
                      f'<title>{html.escape(title)}</title>',
                      f'<desc>{html.escape(subtitle)}</desc>',
                      f'<rect width="1000" height="{height}" fill="white"/>']
        self.image = self.draw = None
        if png:
            from PIL import Image, ImageDraw
            self.image = Image.new('RGB', (1000, height), 'white')
            self.draw = ImageDraw.Draw(self.image)
        self.text(40, 48, title, 29, bold=True)
        self.text(40, 82, subtitle, 17, color=MUTED)

    def font(self, size, bold):
        from PIL import ImageFont
        names = ([r'C:/Windows/Fonts/arialbd.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']
                 if bold else [r'C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'])
        for name in names:
            if Path(name).exists():
                return ImageFont.truetype(name, size)
        return ImageFont.load_default(size=size)

    def text(self, x, y, label, size=20, center=False, bold=False, color=INK):
        label = str(label)
        anchor = 'middle' if center else 'start'
        weight = '700' if bold else '400'
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{html.escape(label)}</text>')
        if self.draw:
            self.draw.text((x, y), label, font=self.font(size, bold), fill=color, anchor='ms' if center else 'ls')

    def box(self, x, y, w, h, lines, fill=BLUE, selected=False, size=20):
        stroke = '#bd7619' if selected else '#b5c6d6'
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{3 if selected else 1.5}"/>')
        if self.draw:
            self.draw.rounded_rectangle((x, y, x+w, y+h), radius=12, fill=fill, outline=stroke, width=3 if selected else 2)
        gap = size + 9
        start = y + h/2 - (len(lines)-1)*gap/2 + size*.34
        for index, line in enumerate(lines):
            self.text(x+w/2, start+index*gap, line, size, center=True, bold=index==0)

    def arrow(self, points, color=INK):
        coords = ' '.join(f'{x},{y}' for x,y in points)
        self.parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')
        x,y = points[-1]
        px,py = points[-2]
        angle = math.atan2(y-py,x-px)
        triangle = [(x,y), (x-11*math.cos(angle-.42),y-11*math.sin(angle-.42)),
                    (x-11*math.cos(angle+.42),y-11*math.sin(angle+.42))]
        self.parts.append('<polygon points="'+' '.join(f'{a:.2f},{b:.2f}' for a,b in triangle)+f'" fill="{color}"/>')
        if self.draw:
            self.draw.line(points, fill=color, width=3)
            self.draw.polygon(triangle, fill=color)

    def svg(self):
        return '\n'.join(self.parts+['</svg>'])+'\n'


def figures(png=False):
    atlas = json.loads((ROOT/'docs/knowledge-bank/data/atlas.json').read_text())
    c = Canvas(760, 'Tiny Aya from text to next-token probabilities', 'Checkpoint dimensions and operations; a schematic, not a measured execution trace.', png)
    c.box(40,140,240,100,['Text to token IDs','Vocabulary: 262,144'])
    c.box(335,140,300,100,['Embedding lookup E','One 2,048-feature vector'],GREEN)
    c.box(690,140,270,100,['36 decoder blocks','Parallel attention + MLP'])
    c.arrow([(280,190),(335,190)])
    c.arrow([(635,190),(690,190)])
    c.arrow([(825,240),(825,315)])
    c.box(40,315,920,130,['Nine repeated groups of four blocks','Local + RoPE  |  Local + RoPE  |  Local + RoPE  |  Global, no explicit RoPE','Causal attention throughout; local window 4,096; configured context 8,192'],PURPLE,size=19)
    c.arrow([(160,445),(160,535)])
    c.box(40,535,240,100,['Final LayerNorm','Width: 2,048'])
    c.box(335,535,300,100,['Shared output matrix E','262,144 vocabulary scores'],GREEN)
    c.box(690,535,270,100,['Softmax','Next-token probabilities'])
    c.arrow([(280,585),(335,585)])
    c.arrow([(635,585),(690,585)])
    c.text(40,697,'The same stored E supplies input lookup and output projection; runtime copies may differ.',18)
    yield 'model-overview', c

    c = Canvas(900, 'One Tiny Aya block has parallel branches', 'Both branches consume the same normalized incoming state.', png)
    c.box(40,140,250,90,['Incoming residual h','2,048 features'])
    c.box(380,140,270,90,['LayerNorm(h)','Center, scale, learned gain'])
    c.arrow([(290,185),(380,185)])
    c.box(200,350,290,110,['Attention(x)','Mix permitted token states'],BLUE)
    c.box(640,350,290,110,['Gated MLP(x)','Transform each token state'],GREEN)
    c.arrow([(515,230),(515,285),(345,285),(345,350)])
    c.arrow([(515,285),(785,285),(785,350)])
    c.text(530,271,'x',20,bold=True)
    c.box(430,620,180,70,['Add all three'],ORANGE)
    c.arrow([(345,460),(345,550),(470,550),(470,620)])
    c.arrow([(785,460),(785,550),(570,550),(570,620)])
    c.arrow([(165,230),(165,655),(430,655)])
    c.text(185,510,'Residual bypass',16)
    c.arrow([(520,690),(520,750)])
    c.box(350,750,340,75,['h_next = h + Attention(x) + MLP(x)'],PURPLE,size=17)
    c.text(40,866,'The next block receives the combined result. There is one shared input norm in each block.',18)
    yield 'parallel-block', c

    c = Canvas(860, 'Attention projections and grouped query heads', 'Q and K form attention weights; V supplies mixed content; O returns to residual width.', png)
    c.box(350,135,300,75,['Normalized input x: 2,048'])
    for x, title, lines in [(40,'Query Q',['16 heads x 128','W_Q: 2,048 x 2,048']),
                             (370,'Key K',['4 heads x 128','W_K: 512 x 2,048']),
                             (700,'Value V',['4 heads x 128','W_V: 512 x 2,048'])]:
        c.box(x,290,260,110,[title]+lines,GREEN if title=='Value V' else BLUE,size=18)
        c.arrow([(500,210),(500,245),(x+130,245),(x+130,290)])
    c.box(65,490,500,105,['Q/K scores + mask -> softmax','Local blocks apply RoPE to Q and K','Four query heads share each KV head'],PURPLE,size=18)
    c.arrow([(170,400),(170,490)])
    c.arrow([(500,400),(500,490)])
    c.box(650,490,310,105,['Weighted sum of V','One result per query head'],GREEN,size=18)
    c.arrow([(830,400),(830,490)])
    c.arrow([(565,542),(650,542)])
    c.arrow([(805,595),(805,675)])
    c.box(600,675,360,100,['Concatenate and project with O','W_O: 2,048 x 2,048','Output: 2,048 features'],BLUE,size=18)
    c.text(40,824,'All attention is causal. Global blocks omit explicit RoPE; they still receive contextual states.',18)
    yield 'attention', c

    c = Canvas(800, 'The gated feed-forward network', 'The up and gate projections meet in an elementwise product.', png)
    c.box(370,135,260,75,['Normalized input x: 2,048'])
    c.box(55,315,360,110,['Gate projection -> SiLU','W_gate: 11,008 x 2,048','11,008 nonlinear gate values'],PURPLE,size=18)
    c.box(585,315,360,110,['Up projection','W_up: 11,008 x 2,048','11,008 intermediate features'],GREEN,size=18)
    c.arrow([(500,210),(500,255),(235,255),(235,315)])
    c.arrow([(500,255),(765,255),(765,315)])
    c.box(355,505,290,75,['Elementwise product g * u'],ORANGE,size=19)
    c.arrow([(235,425),(235,542),(355,542)])
    c.arrow([(765,425),(765,542),(645,542)])
    c.arrow([(500,580),(500,645)])
    c.box(280,645,440,95,['Down projection: 2,048 x 11,008','Returns 2,048 features to residual addition'],BLUE,size=18)
    c.text(40,781,'Each matrix: 22,544,384 parameters. The gate is not a binary switch.',18)
    yield 'mlp', c

    c = Canvas(1180, 'Where the eight pilot matrices are located', 'All 36 blocks; orange outlines mark selected blocks, not measured importance.', png)
    c.box(40,120,920,75,['One additional selected matrix: shared embedding and output','It is outside the decoder blocks and has no same-shape swap partner here.'],GREEN,selected=True,size=18)
    for row in atlas['layers']:
        block = row['block_index']
        x, y = 40+(block%4)*235, 255+(block//4)*89
        names = {'attn_v':'V','ffn_down':'down','ffn_gate':'gate','ffn_up':'up'}
        selected = ', '.join(names[name] for name in row['selected'])
        c.box(x,y,215,75,[f"Block {block} / layer {block+1}",
                         'Global, no explicit RoPE' if not row['local_rope'] else 'Local + RoPE',
                         'Selected: '+selected if selected else 'No selected matrix'],
              PURPLE if not row['local_rope'] else BLUE,selected=bool(selected),size=14)
    c.text(40,1100,'Global blocks: 3, 7, 11, 15, 19, 23, 27, 31, 35 (code indices).',18)
    c.text(40,1134,'Selection spans roles and depth. It does not prove language localization or the best sites.',18)
    yield 'selection-map', c

    c = Canvas(920, 'What the proposed intervention study would measure', 'Design only: comparative C8 measurements have not been collected; execution is paused.', png)
    c.box(40,140,290,105,['BF16 reference','All original stored matrices'],BLUE,size=19)
    c.box(420,140,540,105,['One matrix changed per diagnostic','Eight sites x two actual native formats = 16 files','All other BF16 tensor payloads preserved'],GREEN,size=18)
    c.arrow([(330,192),(420,192)])
    c.arrow([(690,245),(690,305)])
    c.box(40,305,920,110,['Measure prediction damage on independent probe sources','Compare real native KL with separate construction-sample proxies','Repeat 1 and repeat 2: compare English and Bangla precision preferences'],BLUE,size=18)
    c.arrow([(500,415),(500,475)])
    c.box(100,475,800,95,['Freeze one qualifying pair, or explicitly none','A numerical preference screen is not a task-quality claim.'],ORANGE,size=19)
    c.arrow([(500,570),(500,630)])
    c.box(40,630,920,110,['If nominated: export two complete maps with equal actual bytes','Swap higher/lower precision within the pair; lock all other operating-model types','Confirm the opposite language preferences on repetition 3'],PURPLE,size=18)
    c.arrow([(500,740),(500,800)])
    c.box(100,800,800,75,['Decide whether a larger task and generation study is justified'],GREEN,size=19)
    yield 'intervention-study', c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--png', action='store_true')
    args = parser.parse_args()
    if args.check and args.png:
        parser.error('--check does not rewrite PNG previews')
    count = 0
    for name, canvas in figures(args.png):
        payload = canvas.svg().encode()
        path = OUT/(name+'.svg')
        if args.check:
            if not path.exists() or path.read_bytes() != payload:
                raise ValueError('Diagram is stale: '+name)
        else:
            OUT.mkdir(parents=True,exist_ok=True)
            path.write_bytes(payload)
            if args.png:
                canvas.image.save(OUT/(name+'.png'))
        count += 1
    print(json.dumps(dict(status='verified' if args.check else 'generated', diagrams=count,model_execution=False)))


if __name__ == '__main__':
    main()
