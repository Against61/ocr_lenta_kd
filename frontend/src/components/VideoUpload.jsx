import { UploadCloud, Video, X } from 'lucide-react';
import { useRef, useState } from 'react';

const ACCEPTED_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
const ACCEPTED_EXTENSIONS = ['mp4', 'mov', 'avi'];

const isAcceptedVideo = (file) => {
  const extension = file.name.split('.').pop()?.toLowerCase();
  return ACCEPTED_TYPES.includes(file.type) || ACCEPTED_EXTENSIONS.includes(extension);
};

export default function VideoUpload({ file, isProcessing, onFileSelected, onReset }) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState('');

  const selectFile = (nextFile) => {
    if (!nextFile) return;

    if (!isAcceptedVideo(nextFile)) {
      setError('Загрузите видео в формате MP4, MOV или AVI.');
      return;
    }

    setError('');
    onFileSelected(nextFile);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  };

  return (
    <section className="panel upload-panel" aria-labelledby="upload-title">
      <div className="section-heading">
        <p className="eyebrow">Входное видео</p>
        <h2 id="upload-title">Загрузите видео с полками</h2>
        <p>
          Перетащите видео с маршрутом робота, чтобы запустить mock ML-пайплайн:
          извлечение кадров, поиск ценников, OCR и сборку CSV.
        </p>
      </div>

      <div
        className={`dropzone ${isDragging ? 'is-dragging' : ''} ${file ? 'has-file' : ''}`}
        onDragEnter={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo"
          onChange={(event) => selectFile(event.target.files?.[0])}
          hidden
        />

        <div className="dropzone-icon">
          {file ? <Video size={30} /> : <UploadCloud size={32} />}
        </div>

        <div className="dropzone-copy">
          <strong>{file ? file.name : 'Перетащите видео сюда'}</strong>
          <span>{file ? `Выбрано ${(file.size / 1024 / 1024).toFixed(1)} МБ` : 'Поддерживаются MP4, MOV, AVI'}</span>
        </div>

        <div className="upload-actions">
          <button
            className="button primary"
            type="button"
            disabled={isProcessing}
            onClick={() => inputRef.current?.click()}
          >
            Выбрать файл
          </button>
          {file && (
            <button className="icon-button" type="button" disabled={isProcessing} onClick={onReset} aria-label="Удалить файл">
              <X size={18} />
            </button>
          )}
        </div>
      </div>

      {error && <p className="form-error">{error}</p>}
    </section>
  );
}
