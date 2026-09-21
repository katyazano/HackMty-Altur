import React from 'react';

interface FooterProps {
  onNavigate?: (route: 'home' | 'engine1' | 'engine2') => void;
}

export const Footer: React.FC<FooterProps> = ({ onNavigate }) => {
  const handleNav = (route: 'home' | 'engine1' | 'engine2', e: React.MouseEvent) => {
    if (onNavigate) {
      e.preventDefault();
      onNavigate(route);
    }
  };

  return (
    <footer className="site-footer">
      <div className="footer-main-row">
        {/* Column 1: Brand & Hackathon Scope */}
        <div className="footer-col">
          <div className="footer-brand">merge-<span>conflict</span></div>
          <p className="footer-desc">
            High-performance voice biometrics defense engineered for HackMTY. Frontline acoustic filtering combined with multimodal deepfake representation learning.
          </p>
        </div>

        {/* Column 2: Architecture & Endpoints */}
        <div className="footer-col">
          <div className="footer-heading">Architecture</div>
          <div className="footer-link-list">
            <a
              href="#/engine1"
              onClick={(e) => handleNav('engine1', e)}
              className="footer-link"
            >
              <span>Engine 1: Baseline (~700ms)</span>
              <i className="fa-solid fa-arrow-right text-[9px] text-[#7B8290]"></i>
            </a>
            <a
              href="#/engine2"
              onClick={(e) => handleNav('engine2', e)}
              className="footer-link"
            >
              <span>Engine 2: SOTA Multimodal</span>
              <i className="fa-solid fa-arrow-right text-[9px] text-[#7B8290]"></i>
            </a>
            <a
              href="/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-link"
            >
              <span>FastAPI OpenAPI Schema</span>
              <i className="fa-solid fa-arrow-up-right-from-square text-[9px] text-[#7B8290]"></i>
            </a>
          </div>
        </div>

        {/* Column 3: Project & Team Links */}
        <div className="footer-col">
          <div className="footer-heading">Contributors</div>
          <div className="footer-link-list">
            <a
              href="https://github.com/pontro"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-link"
            >
              <i className="fa-brands fa-github text-[#7B8290]"></i>
              <span>pontro</span>
            </a>
            <a
              href="https://github.com/Chewbaccas"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-link"
            >
              <i className="fa-brands fa-github text-[#7B8290]"></i>
              <span>Chewbaccas</span>
            </a>
            <a
              href="https://github.com/katyazano"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-link"
            >
              <i className="fa-brands fa-github text-[#7B8290]"></i>
              <span>katyazano</span>
            </a>
            <a
              href="https://github.com/cris-hernandezz"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-link"
            >
              <i className="fa-brands fa-github text-[#7B8290]"></i>
              <span>cris-hernandezz</span>
            </a>
          </div>
        </div>
      </div>

      {/* Sub-Footer Strip */}
      <div className="footer-bottom-bar">
        <div>merge-conflict // Altur Voice Anti-Spoofing Challenge</div>
        <div>Built for HackMTY 2026</div>
      </div>
    </footer>
  );
};
