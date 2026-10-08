// Gera relatorio.docx a partir de relatorio.md (fonte única do texto).
// Suporta o subconjunto de Markdown usado no relatório: título (#), seções (##), parágrafos, listas (-),
// tabelas (|), imagens ![alt](arquivo){width=Xcm} (várias na mesma linha ficam lado a lado),
// **negrito**, _itálico_ e `código`.
// Uso: node gerar_relatorio.js [relatorio.md] [relatorio.docx]
// Padrão: lê relatorio/relatorio.md e grava relatorio/relatorio.docx (relativos a este arquivo; roda de qualquer
// pasta). Os caminhos das imagens no .md são relativos ao próprio .md (ex.: ../graficos/1_histogramas.png).

const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle,
} = require('docx');

const ENTRADA = process.argv[2] || path.join(__dirname, 'relatorio', 'relatorio.md');
const SAIDA = process.argv[3] || path.join(__dirname, 'relatorio', 'relatorio.docx');
const DXA_POR_CM = 567;
const LARGURA_UTIL = 11906 - 2 * 1134; // A4 com margens de 2 cm
const FONTE = 'Arial';
const CINZA = 'F0EFEC';

// "**a** _b_ `c`" -> TextRuns
function inline(texto, base = {}) {
  const runs = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|_[^_]+_)/g;
  let i = 0;
  for (const m of texto.matchAll(re)) {
    if (m.index > i) runs.push(new TextRun({ text: texto.slice(i, m.index), ...base }));
    const t = m[0];
    if (t.startsWith('**')) runs.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith('`')) runs.push(new TextRun({ text: t.slice(1, -1), font: 'Consolas', size: 18, ...base }));
    else runs.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    i = m.index + t.length;
  }
  if (i < texto.length) runs.push(new TextRun({ text: texto.slice(i), ...base }));
  return runs;
}

// Dimensões de um PNG lidas do cabeçalho IHDR
function dimPng(arq) {
  const b = fs.readFileSync(arq);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), dados: b };
}

function imagens(linha) {
  const runs = [];
  for (const m of linha.matchAll(/!\[([^\]]*)\]\(([^)]+)\)(\{width=([\d.]+)cm\})?/g)) {
    const { w, h, dados } = dimPng(path.resolve(path.dirname(ENTRADA), m[2]));
    const larguraPx = (parseFloat(m[4] || '16') / 2.54) * 96; // docx-js usa pixels a 96 dpi
    if (runs.length) runs.push(new TextRun({ text: '  ' }));
    runs.push(new ImageRun({ type: 'png', data: dados, transformation: { width: larguraPx, height: larguraPx * h / w },
      altText: { title: m[1], description: m[1], name: m[1] } }));
  }
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, children: runs });
}

function tabela(linhas) {
  const celulas = linhas.filter((l) => !/^\|\s*-/.test(l))
    .map((l) => l.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim()));
  const nCol = celulas[0].length;
  const primeira = Math.round(LARGURA_UTIL * (nCol > 5 ? 0.27 : 0.24));
  const resto = Math.floor((LARGURA_UTIL - primeira) / (nCol - 1));
  const larguras = [LARGURA_UTIL - resto * (nCol - 1), ...Array(nCol - 1).fill(resto)];
  const borda = { style: BorderStyle.SINGLE, size: 4, color: 'D0CFCA' };
  return new Table({
    width: { size: LARGURA_UTIL, type: WidthType.DXA },
    columnWidths: larguras,
    rows: celulas.map((linha, r) => new TableRow({
      tableHeader: r === 0,
      cantSplit: true,
      children: linha.map((c, k) => new TableCell({
        width: { size: larguras[k], type: WidthType.DXA },
        shading: r === 0 ? { type: ShadingType.CLEAR, color: 'auto', fill: CINZA } : undefined,
        borders: { top: borda, bottom: borda, left: borda, right: borda },
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        children: [new Paragraph({
          keepNext: r < celulas.length - 1, // mantém a tabela inteira na mesma página
          alignment: k === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
          children: inline(c, { size: 17, bold: r === 0 ? true : undefined }),
        })],
      })),
    })),
  });
}

const md = fs.readFileSync(ENTRADA, 'utf8').replace(/\r/g, '').split('\n');
const corpo = [];
let par = [];
// keepNext: rótulo logo antes de um gráfico fica na mesma página que ele
const fecharPar = (keepNext = false) => {
  if (par.length) corpo.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 },
    keepNext, children: inline(par.join(' ')) }));
  par = [];
};
const proximaEhImagem = (i) => {
  let j = i + 1;
  while (j < md.length && !md[j].trim()) j++;
  return j < md.length && md[j].startsWith('![');
};

for (let i = 0; i < md.length; i++) {
  const l = md[i];
  if (!l.trim()) { fecharPar(proximaEhImagem(i)); continue; }
  if (l.startsWith('# ')) { fecharPar(); corpo.push(new Paragraph({ heading: HeadingLevel.TITLE, children: inline(l.slice(2)) })); continue; }
  if (l.startsWith('## ')) { fecharPar(); corpo.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: inline(l.slice(3)) })); continue; }
  if (l.startsWith('### ')) { fecharPar(); corpo.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: inline(l.slice(4)) })); continue; }
  if (l.startsWith('![')) { fecharPar(); corpo.push(imagens(l)); continue; }
  if (l.startsWith('|')) {
    fecharPar();
    const bloco = [];
    while (i < md.length && md[i].startsWith('|')) bloco.push(md[i++]);
    i--;
    corpo.push(tabela(bloco));
    corpo.push(new Paragraph({ spacing: { after: 60 }, children: [] }));
    continue;
  }
  if (l.startsWith('- ')) {
    fecharPar();
    corpo.push(new Paragraph({ numbering: { reference: 'marcadores', level: 0 }, alignment: AlignmentType.JUSTIFIED,
      spacing: { after: 60 }, children: inline(l.slice(2)) }));
    continue;
  }
  if (/^_.*_$/.test(l.trim()) && par.length === 0) { // legenda / subtítulo
    corpo.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 140 },
      children: [new TextRun({ text: l.trim().slice(1, -1), italics: true, size: 17, color: '52514E' })] }));
    continue;
  }
  par.push(l.trim());
}
fecharPar();

const doc = new Document({
  creator: 'Avaliação 1',
  title: 'Previsão de preços de imóveis em Belo Horizonte',
  styles: {
    default: { document: { run: { font: FONTE, size: 20 }, paragraph: { spacing: { line: 250 } } } },
    paragraphStyles: [
      { id: 'Title', name: 'Title', basedOn: 'Normal', run: { font: FONTE, size: 26, bold: true, color: '000000' },
        paragraph: { spacing: { after: 160 }, alignment: AlignmentType.LEFT } },
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: FONTE, size: 23, bold: true, color: '000000' },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 0, keepNext: true } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: FONTE, size: 21, bold: true }, paragraph: { spacing: { before: 120, after: 60 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: 'marcadores', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•',
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 220 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    children: corpo,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(SAIDA, buf);
  console.log(`-> ${SAIDA}`);
});
