import React, { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { HomePage } from './pages/HomePage';
import { Engine1Page } from './pages/Engine1Page';
import { Engine2Page } from './pages/Engine2Page';

type Route = 'home' | 'engine1' | 'engine2';

export const App: React.FC = () => {
  const getRouteFromUrl = (): Route => {
    const hash = window.location.hash.toLowerCase();
    const pathname = window.location.pathname.toLowerCase();

    if (hash.includes('engine1') || pathname.endsWith('/engine1') || pathname.endsWith('/engine1.html')) {
      return 'engine1';
    }
    if (hash.includes('engine2') || pathname.endsWith('/engine2') || pathname.endsWith('/engine2.html')) {
      return 'engine2';
    }
    return 'home';
  };

  const [currentRoute, setCurrentRoute] = useState<Route>(getRouteFromUrl);

  useEffect(() => {
    const handlePopState = () => {
      setCurrentRoute(getRouteFromUrl());
    };

    window.addEventListener('hashchange', handlePopState);
    window.addEventListener('popstate', handlePopState);
    return () => {
      window.removeEventListener('hashchange', handlePopState);
      window.removeEventListener('popstate', handlePopState);
    };
  }, []);

  const navigate = (route: Route) => {
    setCurrentRoute(route);
    if (route === 'home') {
      window.location.hash = '#/';
    } else {
      window.location.hash = `#/${route}`;
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="app-frame">
      <Navbar currentRoute={currentRoute} onNavigate={navigate} />

      <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {currentRoute === 'home' && <HomePage onNavigate={navigate} />}
        {currentRoute === 'engine1' && <Engine1Page onNavigate={navigate} />}
        {currentRoute === 'engine2' && <Engine2Page onNavigate={navigate} />}
      </main>

      <Footer onNavigate={navigate} />
    </div>
  );
};
