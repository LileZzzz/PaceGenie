import { useState, useEffect } from 'react';
import { Dashboard } from './components/dashboard/Dashboard';
import { ChatInterface } from './components/chat/ChatInterface';
import { Footprints, Menu, ChevronLeft, ChevronRight } from 'lucide-react';
import type { MockGarminData } from './types';
import { API_ENDPOINTS } from '@/lib/config';

// Import mock data
import mockData from '../data/mock_garmin.json';

function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [data] = useState<MockGarminData>(mockData as MockGarminData);
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    fetch(API_ENDPOINTS.health)
      .then(r => setBackendOnline(r.ok))
      .catch(() => setBackendOnline(false));
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation Bar */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass border-b border-border/50">
        <div className="flex items-center justify-between px-4 h-14">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-red-500 to-red-700 flex items-center justify-center">
                <Footprints className="w-5 h-5 text-white" />
              </div>
              <div className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 bg-emerald-500 rounded-full border-2 border-card animate-pulse" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-lg font-bold gradient-text">PaceGenie</h1>
              <p className="text-[10px] text-muted-foreground -mt-0.5">AI Running Assistant</p>
            </div>
          </div>

          {/* Center - Toggle button for desktop */}
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-muted-foreground hover:text-white transition-colors text-sm"
          >
            {sidebarCollapsed ? (
              <>
                <ChevronRight className="w-4 h-4" />
                <span>Expand</span>
              </>
            ) : (
              <>
                <ChevronLeft className="w-4 h-4" />
                <span>Collapse</span>
              </>
            )}
          </button>

          {/* Right actions */}
          <div className="flex items-center gap-1">
            <button
              className="w-9 h-9 flex items-center justify-center rounded-lg hover:bg-white/5 text-muted-foreground hover:text-white transition-colors lg:hidden"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            >
              <Menu className="w-4 h-4" />
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content - Split Screen */}
      <div className="pt-14 h-screen flex">
        {/* Left Panel - Dashboard */}
        <div 
          className={`transition-all duration-500 ease-out overflow-hidden ${
            sidebarCollapsed 
              ? 'w-0 opacity-0' 
              : 'w-full lg:w-[45%] xl:w-[40%] opacity-100'
          }`}
        >
          <div className="h-full overflow-hidden">
            <Dashboard data={data} />
          </div>
        </div>

        {/* Divider */}
        {!sidebarCollapsed && (
          <div className="w-px bg-border/50 hidden lg:block" />
        )}

        {/* Right Panel - Chat */}
        <div 
          className={`transition-all duration-500 ease-out ${
            sidebarCollapsed 
              ? 'w-full' 
              : 'w-full lg:w-[55%] xl:w-[60%]'
          }`}
        >
          <div className="h-full p-4">
            <ChatInterface userId={data.user_id} sessionId={sessionId} backendOnline={backendOnline} />
          </div>
        </div>
      </div>

      {/* Mobile toggle button */}
      <button
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        className="fixed bottom-4 right-4 w-12 h-12 rounded-full shadow-lg bg-gradient-to-br from-red-500 to-red-700 text-white flex items-center justify-center lg:hidden z-50"
      >
        {sidebarCollapsed ? <Footprints className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
      </button>
    </div>
  );
}

export default App;
