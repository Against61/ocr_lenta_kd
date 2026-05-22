import { useEffect, useState } from 'react';
import Header from './components/Header.jsx';
import VideoUpload from './components/VideoUpload.jsx';
import ResultsTable from './components/ResultsTable.jsx';
import CsvDownloadButton from './components/CsvDownloadButton.jsx';
import ProcessingTimeline from './components/ProcessingTimeline.jsx';
import { createVideoJob, getVideoJob, getVideoJobCsv } from './utils/api.js';
import { parseCsvRows } from './utils/csv.js';

const STAGES = [
  { label: 'Загрузка видео', description: 'Передаем файл на сервер обработки.' },
  { label: 'Извлечение кадров', description: 'Читаем видео и выбираем кадры для анализа.' },
  { label: 'Поиск ценников', description: 'Находим и трекаем ценники на кадрах.' },
  { label: 'Распознавание текста и кодов', description: 'Отправляем найденные кропы в OCR/VLM и сканируем коды.' },
  { label: 'Формирование CSV', description: 'Собираем распознанные поля в итоговый CSV.' },
  { label: 'Готово', description: 'Структурированный датасет ценников готов.' },
];

const STAGE_DURATION_MS = 5000;
const JOB_POLL_INTERVAL_MS = 5000;
const VLM_STAGE_INDEX = 3;
const CSV_STAGE_INDEX = 4;

export default function App() {
  const [file, setFile] = useState(null);
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
    setCurrentStageIndex(0);

    const controller = new AbortController();
    let pollTimeout = null;
    const interval = window.setInterval(() => {
      setCurrentStageIndex((previousStage) => Math.min(previousStage + 1, VLM_STAGE_INDEX));
    }, STAGE_DURATION_MS);

    const finishWithCsv = (csv) => {
      if (controller.signal.aborted) return;

      setCurrentStageIndex(CSV_STAGE_INDEX);
      setCsvContent(csv);
      setRows(parseCsvRows(csv));
      window.setTimeout(() => setCurrentStageIndex(STAGES.length - 1), 300);
    };

    const handleRequestError = (requestError) => {
      if (controller.signal.aborted) return;

      setRows([]);
      setCsvContent('');
      setError(requestError.message || 'Не удалось обработать видео');
      setCurrentStageIndex(0);
    };

    const waitForJobCsv = (jobId) =>
      new Promise((resolve, reject) => {
        const pollJob = async () => {
          try {
            const job = await getVideoJob(jobId, controller.signal);
            if (controller.signal.aborted) return;

            if (job.status === 'completed') {
              resolve(await getVideoJobCsv(jobId, controller.signal));
              return;
            }

            if (job.status === 'failed') {
              reject(new Error(job.error || 'Не удалось обработать видео'));
              return;
            }

            pollTimeout = window.setTimeout(pollJob, JOB_POLL_INTERVAL_MS);
          } catch (requestError) {
            reject(requestError);
          }
        };

        pollJob();
      });

    createVideoJob(file, controller.signal)
      .then((job) => {
        if (controller.signal.aborted) return;

        setCurrentStageIndex(1);
        return waitForJobCsv(job.job_id);
      })
      .then((csv) => {
        if (csv) finishWithCsv(csv);
      })
      .catch(handleRequestError)
      .finally(() => {
        if (!controller.signal.aborted) {
          window.clearInterval(interval);
        }
      });

    return () => {
      controller.abort();
      if (pollTimeout !== null) {
        window.clearTimeout(pollTimeout);
      }
      window.clearInterval(interval);
    };
  }, [file]);

  const resetWorkflow = () => {
    setFile(null);
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
          {(file || error) && (
            <ProcessingTimeline
              stages={STAGES}
              currentStageIndex={currentStageIndex}
              isComplete={isComplete}
              hasFile={Boolean(file)}
            />
          )}
        </section>

        {(isProcessing || error) && (
          <section className={`request-status ${error ? 'error' : ''}`} aria-live="polite">
            {error ? error : `${STAGES[currentStageIndex]?.label || 'Обработка видео'}...`}
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
