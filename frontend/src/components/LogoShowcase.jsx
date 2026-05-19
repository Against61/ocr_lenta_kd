import { motion } from 'framer-motion';

const LOGO_SRC = 'https://kozhindev.com/images/covers/default.webp';

export default function LogoShowcase() {
  return (
    <motion.div
      className="logo-showcase"
      aria-label="KozhinDev logo centerpiece"
      initial={{ opacity: 0, y: 18, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.8, ease: 'easeOut' }}
      whileHover={{ y: -6, scale: 1.018 }}
    >
      <span className="logo-ambient logo-ambient-cyan" aria-hidden="true" />
      <span className="logo-ambient logo-ambient-purple" aria-hidden="true" />

      <motion.span
        className="logo-orbit"
        aria-hidden="true"
        animate={{ rotate: 360 }}
        transition={{ duration: 32, repeat: Infinity, ease: 'linear' }}
      />
      <motion.span
        className="logo-orbit logo-orbit-secondary"
        aria-hidden="true"
        animate={{ rotate: -360 }}
        transition={{ duration: 44, repeat: Infinity, ease: 'linear' }}
      />

      <motion.span
        className="logo-core"
        animate={{
          boxShadow: [
            '0 42px 110px rgba(45,212,191,0.18), inset 0 1px 0 rgba(255,255,255,0.14)',
            '0 52px 140px rgba(168,85,247,0.22), inset 0 1px 0 rgba(255,255,255,0.18)',
            '0 42px 110px rgba(45,212,191,0.18), inset 0 1px 0 rgba(255,255,255,0.14)',
          ],
        }}
        transition={{ duration: 5.8, repeat: Infinity, ease: 'easeInOut' }}
      >
        <span className="logo-core-grid" aria-hidden="true" />
        <span className="logo-reflection" aria-hidden="true" />
        <img src={LOGO_SRC} alt="Логотип KozhinDev" />
      </motion.span>
    </motion.div>
  );
}
