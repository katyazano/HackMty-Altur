import React from 'react';

interface NavbarProps {
  currentRoute?: 'home' | 'engine1' | 'engine2';
  onNavigate?: (route: 'home' | 'engine1' | 'engine2') => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentRoute = 'home',
  onNavigate,
}) => {
  const handleNav = (route: 'home' | 'engine1' | 'engine2', e: React.MouseEvent) => {
    if (onNavigate) {
      e.preventDefault();
      onNavigate(route);
    }
  };

  return (
    <header className="top-navbar-grid">
      {/* Left Spacer / Badge Grid Wing */}
      <div className="nav-cell nav-cell-spacer-left">
        <div
          className="mono text-xs text-[#555B67] flex items-center gap-2"
          style={{
            fontSize: '11px',
            fontFamily: "'JetBrains Mono', monospace",
            color: '#555B67',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <span
            style={{
              width: '7px',
              height: '7px',
              borderRadius: '50%',
              backgroundColor: '#FF5500',
              display: 'inline-block',
            }}
          />
          <span style={{ textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 500 }}>
            Altur HackMTY Challenge
          </span>
        </div>
      </div>

      {/* Left of mergeConflict: Engine 1 Square Cell */}
      <a
        href="#/engine1"
        onClick={(e) => handleNav('engine1', e)}
        className={`nav-engine-cell ${currentRoute === 'engine1' ? 'active' : ''}`}
        title="Engine 1 Documentation"
      >
        <span className="nav-engine-title">Engine 1</span>
        <span className="nav-engine-sub">docs</span>
      </a>

      {/* Center: "mergeConflict" in the VERY CENTER */}
      <div className="nav-cell center-nav-box">
        <a
          href="#/"
          onClick={(e) => handleNav('home', e)}
          className="center-title"
          title="mergeConflict Home"
        >
          merge<span>Conflict</span>
        </a>
      </div>

      {/* Right of mergeConflict: Engine 2 Square Cell */}
      <a
        href="#/engine2"
        onClick={(e) => handleNav('engine2', e)}
        className={`nav-engine-cell ${currentRoute === 'engine2' ? 'active' : ''}`}
        title="Engine 2 Documentation"
      >
        <span className="nav-engine-title">Engine 2</span>
        <span className="nav-engine-sub">docs</span>
      </a>

      {/* Right Spacer Grid Wing with Project GitHub Repo Link */}
      <div className="nav-cell nav-cell-spacer-right">
        <a
          href="https://github.com/katyazano/HackMty-Altur"
          target="_blank"
          rel="noopener noreferrer"
          className="nav-github-link"
          title="GitHub Repository"
        >
          <i className="fa-brands fa-github text-base"></i>
          <span>GitHub</span>
          <i className="fa-solid fa-arrow-up-right-from-square text-[9px] text-[#7B8290]"></i>
        </a>
      </div>
    </header>
  );
};
