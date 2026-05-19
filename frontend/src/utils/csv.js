const CSV_COLUMNS = [
  'frame_id',
  'timestamp',
  'shelf_id',
  'price_tag_id',
  'product_name',
  'price',
  'confidence',
];

const escapeCsvValue = (value) => {
  const normalizedValue = value === null || value === undefined ? '' : String(value);
  return /[",\n]/.test(normalizedValue)
    ? `"${normalizedValue.replaceAll('"', '""')}"`
    : normalizedValue;
};

export const convertRowsToCsv = (rows) => {
  const header = CSV_COLUMNS.join(',');
  const body = rows
    .map((row) => CSV_COLUMNS.map((column) => escapeCsvValue(row[column])).join(','))
    .join('\n');

  return `${header}\n${body}`;
};

export const downloadCsv = (rows, filename = 'price-tag-results.csv') => {
  const csv = convertRowsToCsv(rows);
  downloadCsvText(csv, filename);
};

export const downloadCsvText = (csv, filename = 'price-tag-results.csv') => {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');

  link.href = url;
  link.download = filename;
  link.click();

  URL.revokeObjectURL(url);
};

const parseCsvLine = (line) => {
  const values = [];
  let currentValue = '';
  let isQuoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    const nextCharacter = line[index + 1];

    if (character === '"' && isQuoted && nextCharacter === '"') {
      currentValue += '"';
      index += 1;
      continue;
    }

    if (character === '"') {
      isQuoted = !isQuoted;
      continue;
    }

    if (character === ',' && !isQuoted) {
      values.push(currentValue);
      currentValue = '';
      continue;
    }

    currentValue += character;
  }

  values.push(currentValue);
  return values;
};

export const parseCsvRows = (csv) => {
  const lines = csv
    .trim()
    .split(/\r?\n/)
    .filter(Boolean);

  if (!lines.length) return [];

  const headers = parseCsvLine(lines[0]);

  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    return headers.reduce((row, header, index) => {
      row[header] = values[index] ?? '';
      return row;
    }, {});
  });
};
