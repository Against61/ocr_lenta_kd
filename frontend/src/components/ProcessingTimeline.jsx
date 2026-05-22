import { Check, Circle, Loader2 } from 'lucide-react';

export default function ProcessingTimeline({ stages, currentStageIndex, isComplete, hasFile }) {
  const currentLabel = hasFile ? (isComplete ? 'Готово к скачиванию' : stages[currentStageIndex]?.label) : 'Ожидаем видео';

  return (
    <section className="panel processing-panel" aria-labelledby="processing-title">
      <div className="section-heading compact">
        <p className="eyebrow">ML-пайплайн</p>
        <h2 id="processing-title">Статус обработки</h2>
        <p>{currentLabel}</p>
      </div>

      <ol className="timeline" aria-label="Этапы обработки видео">
        {stages.map((stage, index) => {
          const isDone = index < currentStageIndex || isComplete;
          const isActive = index === currentStageIndex && hasFile && !isComplete;

          return (
            <li className={`timeline-item ${isDone ? 'done' : ''} ${isActive ? 'active' : ''}`} key={stage.label}>
              <span className="timeline-icon">
                {isDone && <Check size={16} />}
                {isActive && <Loader2 size={16} />}
                {!isDone && !isActive && <Circle size={14} />}
              </span>
              <span>
                <strong>{stage.label}</strong>
                <small>{stage.description}</small>
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
