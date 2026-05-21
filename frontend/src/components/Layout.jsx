import { Outlet, Link, useLocation } from 'react-router-dom';
import { Database, Activity, LayoutDashboard, Settings, BarChart2, Clock, LineChart } from 'lucide-react';
import { motion } from 'framer-motion';

const navItems = [
  { path: '/', label: 'Overview', icon: LayoutDashboard },
  { path: '/dataset', label: 'Dataset', icon: Database },
  { path: '/experiments', label: 'Train Model', icon: Activity },
  { path: '/history', label: 'Experiments', icon: Clock },
  { path: '/compare', label: 'Model Comparison', icon: BarChart2 },
  { path: '/drift', label: 'Data Drift', icon: LineChart },
];

export default function Layout() {
  const location = useLocation();

  return (
    <div className="flex h-screen bg-surface-100 text-text-800 overflow-hidden selection:bg-brand-100 selection:text-brand-600">
      {/* Sidebar */}
      <motion.aside 
        initial={{ x: -250 }}
        animate={{ x: 0 }}
        className="w-64 border-r border-surface-200 bg-surface-50/60 backdrop-blur-xl flex flex-col flex-shrink-0 z-20 shadow-[-10px_0_30px_-15px_rgba(0,0,0,0.1)_inset]"
      >
        <div className="h-20 flex items-center px-6 border-b border-surface-200/50 bg-transparent">
          <div className="h-10 w-10 bg-brand-50 rounded-xl flex items-center justify-center mr-3 border border-brand-100">
            <Activity size={22} className="text-brand-500" />
          </div>
          <span className="font-extrabold text-2xl tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-text-900 to-brand-600">
            ML Hub
          </span>
        </div>
        
        <nav className="flex-1 px-4 py-8 space-y-2 relative">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path || 
                           (item.path !== '/' && location.pathname.startsWith(item.path));
            return (
              <Link
                key={item.path}
                to={item.path}
                className="block relative group"
              >
                {isActive && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-brand-50 rounded-xl"
                    initial={false}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  />
                )}
                <div className={`relative z-10 flex items-center px-4 py-3 rounded-xl transition-colors duration-200 font-medium ${
                  isActive 
                    ? 'text-brand-600' 
                    : 'text-text-500 hover:text-text-900 hover:bg-surface-100'
                }`}>
                  <item.icon size={20} className={`mr-3 transition-colors ${isActive ? 'text-brand-500' : 'text-text-400 group-hover:text-text-600'}`} />
                  <span>{item.label}</span>
                </div>
              </Link>
            );
          })}
        </nav>
      </motion.aside>

      <main className="flex-1 overflow-auto relative bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMSIgY3k9IjEiIHI9IjEiIGZpbGw9IiNlMmU4ZjAiIGZpbGwtb3BhY2l0eT0iMC40Ii8+PC9zdmc+')]">
        <div className="h-20 border-b border-surface-200/60 bg-surface-50/60 flex items-center justify-between px-8 sticky top-0 z-10 backdrop-blur-xl">
          <div className="flex items-center space-x-4">
             <h1 className="font-extrabold text-2xl text-text-900 tracking-tight flex items-center gap-3">
               {navItems.find(i => i.path === location.pathname)?.label || 'Results Dashboard'}
             </h1>
          </div>
          <div className="flex items-center space-x-3 text-sm font-medium">
             <span className="text-text-500">Project Status:</span>
             <span className="flex items-center text-green-700 bg-green-50 px-2.5 py-1 rounded-full border border-green-200 shadow-sm relative overflow-hidden">
                <span className="absolute inset-0 bg-green-200 opacity-20 animate-pulse"></span>
                <span className="h-2 w-2 bg-green-500 rounded-full mr-2 shadow-[0_0_8px_rgba(34,197,94,0.6)]"></span>
                Online & Tracking
             </span>
          </div>
        </div>
        
        <div className="p-4 pb-12 w-full">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
