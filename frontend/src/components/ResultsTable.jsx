export default function ResultsTable({ rows }) {
  const columns = getVisibleColumns(rows);

  if (!rows.length) {
    return (
      <section className="panel empty-results">
        <p className="eyebrow">Результат</p>
        <h2>Здесь появится preview CSV</h2>
        <p>После обработки извлечённые ценники будут доступны в таблице и в CSV-файле для скачивания.</p>
      </section>
    );
  }

  return (
    <section className="panel results-panel" aria-labelledby="results-title">
      <div className="section-heading compact">
        <p className="eyebrow">Предпросмотр результата</p>
        <h2 id="results-title">Извлечённые ценники</h2>
      </div>

      <div className="table-shell">
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column.key}>{column.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={row.price_tag_id || row.filename || rowIndex}>
                {columns.map((column) => (
                  <td key={column.key}>
                    {column.key === 'confidence' ? <span className="confidence">{row[column.key]}</span> : row[column.key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

const PREFERRED_COLUMNS = [
  { key: 'frame_id', label: 'Кадр' },
  { key: 'timestamp', label: 'Время' },
  { key: 'shelf_id', label: 'Полка' },
  { key: 'price_tag_id', label: 'ID ценника' },
  { key: 'filename', label: 'Файл' },
  { key: 'product_name', label: 'Товар' },
  { key: 'price', label: 'Цена' },
  { key: 'price_default', label: 'Цена' },
  { key: 'price_card', label: 'Цена по карте' },
  { key: 'price_discount', label: 'Скидочная цена' },
  { key: 'barcode', label: 'Штрихкод' },
  { key: 'frame_timestamp', label: 'Время кадра' },
  { key: 'confidence', label: 'Уверенность' },
];

const getVisibleColumns = (rows) => {
  if (!rows.length) return [];

  const availableKeys = Object.keys(rows[0]);
  const preferredColumns = PREFERRED_COLUMNS.filter((column) => availableKeys.includes(column.key));

  if (preferredColumns.length) return preferredColumns.slice(0, 8);

  return availableKeys.slice(0, 8).map((key) => ({ key, label: key }));
};
