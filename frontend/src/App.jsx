import { useEffect, useState } from 'react';
import Header from './components/Header.jsx';
import VideoUpload from './components/VideoUpload.jsx';
import ResultsTable from './components/ResultsTable.jsx';
import CsvDownloadButton from './components/CsvDownloadButton.jsx';
import { analyzeVideo } from './utils/api.js';
import { parseCsvRows } from './utils/csv.js';

const STAGES = [
  { label: 'Загрузка видео', description: 'Подготавливаем видео с маршрутом робота к анализу.' },
  { label: 'Извлечение кадров', description: 'Выбираем кадры из проезда вдоль торговых полок.' },
  { label: 'Поиск ценников', description: 'Находим ценники на полках с помощью моделей компьютерного зрения.' },
  { label: 'Распознавание текста/OCR', description: 'Считываем названия товаров и цены с найденных ценников.' },
  { label: 'Формирование CSV', description: 'Нормализуем строки и оценки уверенности для экспорта.' },
  { label: 'Готово', description: 'Структурированный датасет ценников готов.' },
];

const STAGE_DURATION_MS = 900;

export default function App() {
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [rows, setRows] = useState([]);
  const [csvContent, setCsvContent] = useState('');
  const [error, setError] = useState('');

  const isComplete = Boolean(csvContent);
  const isProcessing = Boolean(file) && !isComplete;

  useEffect(() => {
    if (!file) return undefined;

    setRows([]);
    setCsvContent('');
    setError('');
    setProgress(0);
    setCurrentStageIndex(0);

    const controller = new AbortController();
    const interval = window.setInterval(() => {
      setProgress((previousProgress) => {
        const nextProgress = Math.min(previousProgress + 6, 92);
        const totalSteps = STAGES.length;
        const nextStage = Math.min(Math.floor(nextProgress / (100 / totalSteps)), totalSteps - 1);
        setCurrentStageIndex(nextStage);
        return nextProgress;
      });
    }, STAGE_DURATION_MS);

    analyzeVideo(file, controller.signal)
      .then((csv) => {
        if (controller.signal.aborted) return;

        setCsvContent(csv);
        setRows(parseCsvRows(csv));
        setCurrentStageIndex(STAGES.length - 1);
        setProgress(100);
      })
      .catch((requestError) => {
        if (controller.signal.aborted) return;

        setRows([]);
        setCsvContent('');
        setError(requestError.message || 'Не удалось обработать видео');
        setProgress(0);
        setCurrentStageIndex(0);
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          window.clearInterval(interval);
        }
      });

    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  }, [file]);

  const resetWorkflow = () => {
    setFile(null);
    setProgress(0);
    setCurrentStageIndex(0);
    setRows([]);
    setCsvContent('');
    setError('');
  };

  return (
    <div className="app">
      <div className="background-grid" />
      <Header />

      <main>
        <section className="hero" aria-labelledby="hero-title">
          <div className="hero-content">
            <p className="eyebrow">Компьютерное зрение для ритейла</p>
            <h1 id="hero-title">Превращаем видео полок в датасет ценников.</h1>
            <p>
              Загрузите видео с робота, отслеживайте этапы AI-анализа, извлекайте данные о ценниках
              из кадров и выгружайте готовый CSV для аналитики.
            </p>
            <div className="hero-actions">
              <CsvDownloadButton rows={rows} csvContent={csvContent} disabled={!isComplete} />
            </div>
          </div>

          <div className="hero-visual" aria-hidden="true">
            <div className="scanner-card">
              <div className="scanner-top">
                <span />
                <span />
                <span />
              </div>
              <div className="shelf-frame">
                <div className="scan-line" />
                {Array.from({ length: 9 }).map((_, index) => (
                  <span className="price-box" key={index}>
                    <small>{index % 3 === 0 ? 'OCR' : 'ЦЕННИК'}</small>
                  </span>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="workflow-grid">
          <VideoUpload file={file} isProcessing={isProcessing} onFileSelected={setFile} onReset={resetWorkflow} />
        </section>

        {(isProcessing || error) && (
          <section className={`request-status ${error ? 'error' : ''}`} aria-live="polite">
            {error ? error : `Видео отправлено на анализ. Прогресс обработки: ${Math.round(progress)}%`}
          </section>
        )}

        <ResultsTable rows={rows} />
      </main>

      <footer>
        <span>
          Проект разработан ML отделом компании{' '}
          <a href="https://kozhindev.com" target="_blank" rel="noreferrer">
            KozhinDev
          </a>
        </span>
        <span>Хакатон-демо для анализа ценников</span>
      </footer>
    </div>
  );
}
