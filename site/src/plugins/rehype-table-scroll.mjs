// Wrap every markdown table in a scroll container, and mark the columns that
// must never wrap.
//
// A wide table inside a 45rem column has two bad options in CSS alone: squeeze
// the columns until every cell wraps one word per line, or overflow the page.
// The third option needs an element the markdown does not have, so this plugin
// adds it — `<div class="lk-table-scroll">` — and the stylesheet puts the
// horizontal scroll on that rather than on the body.
//
// The wrapper alone is not enough. `table-layout: auto` hands width to the
// column with the longest content, and these documents mix six columns of
// two-word labels with one column of a paragraph. The labels lose, and
// "OHF-Voice donations" becomes two lines in a 163-pixel column while the
// paragraph beside it gets 400. So the plugin measures each column and marks
// the short ones `lk-nowrap`: they then claim their natural width, the prose
// column takes what is left, and anything still too wide scrolls.
//
// It runs on the parsed HTML tree, after remark, so it only ever sees real
// table elements and never the pipe characters inside a fenced code block.

import { visit } from 'unist-util-visit';

/** A column whose longest cell is at most this many characters cannot wrap. */
const NOWRAP_MAX_CHARS = 30;

/**
 * One figure: "2.16x", "1.79s", "2.1 GB", "20.1", "1.3–1.7x", "255+".
 * A cell is numeric when every slash-separated part is a figure, so
 * "1.79s / 1.84s" counts. An em dash or an empty cell decides nothing.
 */
const FIGURE = /^[0-9][\d.,+]*(\s?[–-]\s?[\d.,]+)?\s?(x|×|s|ms|kb|mb|gb|%)?$/i;

function isNumericCell(text) {
  if (text === '' || text === '—' || text === '-') return true;
  return text.split('/').every((part) => FIGURE.test(part.trim()));
}

function textOf(node) {
  if (node.type === 'text') return node.value;
  if (!node.children) return '';
  return node.children.map(textOf).join('');
}

/** Every `<tr>` in a table, in order, regardless of thead/tbody nesting. */
function rowsOf(table) {
  const rows = [];
  visit(table, 'element', (node) => {
    if (node.tagName === 'tr') rows.push(node);
  });
  return rows;
}

export function rehypeTableScroll() {
  return (tree) => {
    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'table' || !parent || index === null) return;

      // Longest cell per column index, header row included. Alongside it:
      // whether every body cell in the column is a figure (`numeric`, with
      // `hasFigure` guarding all-empty columns), and which column a header
      // literally names "notes".
      const widest = [];
      const numeric = [];
      const hasFigure = [];
      const isNotes = [];
      for (const row of rowsOf(node)) {
        const cells = row.children.filter(
          (c) => c.type === 'element' && (c.tagName === 'td' || c.tagName === 'th'),
        );
        cells.forEach((cell, column) => {
          const text = textOf(cell).trim();
          widest[column] = Math.max(widest[column] ?? 0, text.length);
          if (cell.tagName === 'th') {
            if (text.toLowerCase() === 'notes') isNotes[column] = true;
          } else {
            const figure = isNumericCell(text);
            numeric[column] = (numeric[column] ?? true) && figure;
            if (figure && text !== '' && text !== '—' && text !== '-') {
              hasFigure[column] = true;
            }
          }
        });
      }

      for (const row of rowsOf(node)) {
        const cells = row.children.filter(
          (c) => c.type === 'element' && (c.tagName === 'td' || c.tagName === 'th'),
        );
        cells.forEach((cell, column) => {
          const classes = [];
          if ((widest[column] ?? 0) <= NOWRAP_MAX_CHARS) classes.push('lk-nowrap');
          if (numeric[column] && hasFigure[column]) classes.push('lk-num');
          if (isNotes[column] && cell.tagName === 'td') classes.push('lk-notes');
          if (classes.length === 0) return;
          cell.properties ??= {};
          const existing = cell.properties.className;
          cell.properties.className = Array.isArray(existing)
            ? [...existing, ...classes]
            : classes;
        });
      }

      parent.children[index] = {
        type: 'element',
        tagName: 'div',
        properties: { className: ['lk-table-scroll'] },
        children: [node],
      };
      // The replacement subtree contains the table itself; do not descend
      // into it again.
      return ['skip'];
    });
  };
}
