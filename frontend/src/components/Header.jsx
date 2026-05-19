import { Cpu, Sparkles } from 'lucide-react';
import LogoShowcase from './LogoShowcase.jsx';

const TECH_BADGES = ['Computer Vision', 'OCR', 'CSV Export', 'Retail AI'];

export default function Header() {
  return (
    <header className="header">
      <a className="brand" href="/" aria-label="KozhinDev Анализ ценников">
        <LogoShowcase />
        <span className="brand-copy">
          <span className="brand-name">KozhinDev</span>
          <span className="brand-subtitle">AI Retail Analytics Platform</span>
          <span className="brand-badges" aria-label="Технологии платформы">
            {TECH_BADGES.map((badge) => (
              <span key={badge}>{badge}</span>
            ))}
          </span>
        </span>
      </a>

      <div className="header-actions" aria-label="Статус продукта">
        <span className="status-pill">
          <Sparkles size={16} />
          ML-демо
        </span>
        <span className="status-pill muted">
          <Cpu size={16} />
          Экспорт CSV
        </span>
      </div>
    </header>
  );
}
