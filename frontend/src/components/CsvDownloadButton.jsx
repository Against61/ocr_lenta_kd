import { Download } from 'lucide-react';
import { downloadCsv, downloadCsvText } from '../utils/csv.js';

export default function CsvDownloadButton({ rows, csvContent, disabled }) {
  return (
    <button
      className="button download"
      type="button"
      disabled={disabled || (!rows.length && !csvContent)}
      onClick={() => {
        if (csvContent) {
          downloadCsvText(csvContent);
          return;
        }

        downloadCsv(rows);
      }}
    >
      <Download size={18} />
      Скачать CSV
    </button>
  );
}
