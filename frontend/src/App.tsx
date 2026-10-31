import { useEffect, useMemo, useState } from 'react';
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Moon, SunMedium, UploadCloud, Wallet, ShieldAlert, ArrowRight } from 'lucide-react';
import { api } from './api/client';
import type { AskResponse, SummaryResponse } from './types';

const currency = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 0,
});

const pageItems = ['Dashboard', 'Forecast', 'Anomalies', 'Ask Advisor', 'Ledger'];

function formatInr(value: number) {
  return currency.format(value);
}

function StatCard({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-soft dark:border-slate-700 dark:bg-slate-900">
      <div className="text-xs uppercase tracking-[0.2em] text-slate-500">{label}</div>
      <div className={`mt-3 text-2xl font-bold ${accent ?? 'text-slate-900 dark:text-white'}`}>{value}</div>
    </div>
  );
}

function DashboardPage() {
  const [summary, setSummary] = useState<SummaryResponse>({
    current_balance: 132400,
    inflow_30d: 215000,
    outflow_30d: 180000,
    net_30d: 35000,
    burn_rate: 6200,
    runway_days: 21.3,
    category_breakdown: { 'Sales Revenue': 98000, Rent: 42000, Salary: 68000, Marketing: 18000 },
    monthly_series: [
      { month: '2024-05', inflow: 180000, outflow: 120000, net: 60000 },
      { month: '2024-06', inflow: 210000, outflow: 140000, net: 70000 },
    ],
    risk_status: 'SAFE',
  });
  useEffect(() => {
    api.getSummary('demo').then(setSummary).catch(() => undefined);
  }, []);

  const pieData = useMemo(() => Object.entries(summary.category_breakdown).map(([name, value]) => ({ name, value })), [summary]);

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Current balance" value={formatInr(summary.current_balance)} accent="text-emerald-600" />
        <StatCard label="30d net" value={formatInr(summary.net_30d)} accent="text-sky-600" />
        <StatCard label="Burn rate" value={formatInr(summary.burn_rate)} accent="text-amber-600" />
        <StatCard label="Runway" value={`${summary.runway_days.toFixed(1)} days`} accent="text-violet-600" />
      </div>
      <div className="grid gap-6 xl:grid-cols-[1.3fr_0.7fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Monthly cash flow</h3>
            <span className="rounded-full bg-emerald-100 px-2.5 py-1 text-xs font-semibold text-emerald-700">{summary.risk_status}</span>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={summary.monthly_series}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="month" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip formatter={(value) => formatInr(Number(value))} />
                <Bar dataKey="inflow" fill="#0f766e" radius={[4, 4, 0, 0]} />
                <Bar dataKey="outflow" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Category mix</h3>
          <div className="mt-4 h-52">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={45} outerRadius={70} paddingAngle={4}>
                  {pieData.map((entry, index) => (
                    <Cell key={entry.name} fill={['#0f766e', '#f59e0b', '#2dd4bf', '#94a3b8'][index % 4]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => formatInr(Number(value))} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

function ForecastPage() {
  const [horizon, setHorizon] = useState(30);
  const forecast = useMemo(() => ({
    forecast: [
      { date: '2024-06-01', projected_balance: 125000 },
      { date: '2024-06-15', projected_balance: 118500 },
      { date: '2024-07-01', projected_balance: 132400 },
    ],
    lower_band: 109000,
    upper_band: 140000,
  }), []);
  return (
    <div className="space-y-6">
      <div className="flex space-x-3">
        {[30, 60, 90].map((value) => (
          <button key={value} onClick={() => setHorizon(value)} className={`rounded-full px-4 py-2 text-sm font-medium ${horizon === value ? 'bg-brand-600 text-white' : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200'}`}>
            {value}d
          </button>
        ))}
      </div>
      <div className="rounded-2xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg font-semibold">Projected balance</h3>
          <span className="rounded-full bg-amber-100 px-2.5 py-1 text-xs font-semibold text-amber-700">Model accuracy: 8.5% sMAPE</span>
        </div>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={forecast.forecast}>
              <defs>
                <linearGradient id="balanceFill" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#0f766e" stopOpacity={0.45} />
                  <stop offset="100%" stopColor="#0f766e" stopOpacity={0.08} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip formatter={(value) => formatInr(Number(value))} />
              <Area type="monotone" dataKey="projected_balance" stroke="#0f766e" fill="url(#balanceFill)" strokeWidth={3} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function AnomaliesPage() {
  const anomalies = [
    { date: '2024-06-14', category: 'Marketing', amount: 35000, severity: 'High', reason: 'Above the normal seasonal spend band.' },
    { date: '2024-06-19', category: 'Utilities', amount: 21000, severity: 'Medium', reason: 'Higher than the median run-rate for the month.' },
  ];
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
      <table className="w-full text-left text-sm">
        <thead className="text-slate-500">
          <tr>
            <th className="pb-3">Date</th>
            <th className="pb-3">Category</th>
            <th className="pb-3">Amount</th>
            <th className="pb-3">Severity</th>
            <th className="pb-3">Reason</th>
          </tr>
        </thead>
        <tbody>
          {anomalies.map((item) => (
            <tr key={`${item.date}-${item.category}`} className="border-t border-slate-200 text-slate-700 dark:border-slate-700 dark:text-slate-200">
              <td className="py-3">{item.date}</td>
              <td className="py-3">{item.category}</td>
              <td className="py-3">{formatInr(item.amount)}</td>
              <td className="py-3"><span className={`rounded-full px-2 py-1 text-xs font-semibold ${item.severity === 'High' ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'}`}>{item.severity}</span></td>
              <td className="py-3">{item.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AskAdvisorPage() {
  const [answer] = useState<AskResponse | null>({
    verdict: 'SAFE',
    numbers: { purchase_amount: 80000, safety_buffer: 50000, post_purchase_min_balance: 98000, buffer_delta: 48000 },
    explanation: 'This purchase remains above the required cash buffer with room to absorb payment timing variance.',
    explanation_source: 'template',
    guardrail_passed: true,
  });
  const verdictClasses = answer?.verdict === 'SAFE' ? 'border-emerald-200 bg-emerald-50 text-emerald-800' : answer?.verdict === 'RISKY' ? 'border-amber-200 bg-amber-50 text-amber-800' : 'border-red-200 bg-red-50 text-red-800';

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
        <div className="mb-4 flex flex-wrap gap-2">
          {['Can I afford ₹80,000 in 45 days?', 'What if payroll is 20k higher?', 'Should I hire this quarter?'].map((chip) => (
            <button key={chip} className="rounded-full border border-slate-200 bg-slate-100 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-200 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">{chip}</button>
          ))}
        </div>
        <div className={`inline-flex rounded-full border px-4 py-2 text-sm font-semibold ${verdictClasses}`}>{answer?.verdict}</div>
        <div className="mt-5 grid gap-4 md:grid-cols-2">
          <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800">
            <div className="text-xs uppercase tracking-[0.2em] text-slate-500">Numbers</div>
            <ul className="mt-3 space-y-2 text-sm text-slate-700 dark:text-slate-200">
              <li>Purchase amount: {formatInr(answer?.numbers.purchase_amount ?? 0)}</li>
              <li>Safety buffer: {formatInr(answer?.numbers.safety_buffer ?? 0)}</li>
              <li>Post purchase low: {formatInr(answer?.numbers.post_purchase_min_balance ?? 0)}</li>
            </ul>
          </div>
          <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800">
            <div className="text-xs uppercase tracking-[0.2em] text-slate-500">Explanation source</div>
            <div className="mt-3 inline-flex rounded-full bg-brand-50 px-2.5 py-1 text-xs font-semibold text-brand-700">{answer?.explanation_source === 'llm' ? 'LLM-verified' : 'Template'}</div>
            <p className="mt-4 text-sm text-slate-700 dark:text-slate-200">{answer?.explanation}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function LedgerPage() {
  const rows = [
    { date: '2024-06-25', description: 'Retail sales', amount: 24000, direction: 'inflow', category: 'Sales Revenue' },
    { date: '2024-06-26', description: 'Office rent', amount: 42000, direction: 'outflow', category: 'Rent' },
  ];
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold">Ledger</h3>
        <div className="flex gap-2">
          <button className="rounded-full border border-slate-200 bg-slate-100 px-3 py-2 text-sm font-medium text-slate-700 dark:bg-slate-800 dark:text-slate-200">CSV</button>
          <button className="rounded-full border border-slate-200 bg-slate-100 px-3 py-2 text-sm font-medium text-slate-700 dark:bg-slate-800 dark:text-slate-200">PDF</button>
        </div>
      </div>
      <table className="w-full text-left text-sm">
        <thead className="text-slate-500">
          <tr>
            <th className="pb-3">Date</th>
            <th className="pb-3">Description</th>
            <th className="pb-3">Amount</th>
            <th className="pb-3">Direction</th>
            <th className="pb-3">Category</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.date}-${row.description}`} className="border-t border-slate-200 dark:border-slate-700">
              <td className="py-3">{row.date}</td>
              <td className="py-3">{row.description}</td>
              <td className={`py-3 font-medium ${row.direction === 'inflow' ? 'text-emerald-600' : 'text-red-600'}`}>{formatInr(row.amount)}</td>
              <td className="py-3">{row.direction}</td>
              <td className="py-3">{row.category}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [activePage, setActivePage] = useState('Dashboard');
  const [isServerWarming, setIsServerWarming] = useState(true);

  useEffect(() => {
    const timer = window.setTimeout(() => setIsServerWarming(false), 650);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  const pageMap: Record<string, JSX.Element> = {
    Dashboard: <DashboardPage />,
    Forecast: <ForecastPage />,
    'Anomalies': <AnomaliesPage />,
    'Ask Advisor': <AskAdvisorPage />,
    Ledger: <LedgerPage />,
  };

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <div className="flex min-h-screen flex-col lg:flex-row">
        <aside className="w-full border-b border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900 lg:w-72 lg:border-b-0 lg:border-r">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-600 text-white"><Wallet size={20} /></div>
              <div>
                <div className="text-sm font-semibold">Cashflow</div>
                <div className="text-xs text-slate-500">Risk Assistant</div>
              </div>
            </div>
            <button onClick={() => setDarkMode((value) => !value)} className="rounded-full border border-slate-200 p-2 text-slate-700 dark:border-slate-700 dark:text-slate-200">
              {darkMode ? <SunMedium size={18} /> : <Moon size={18} />}
            </button>
          </div>
          <nav className="mt-8 space-y-2">
            {pageItems.map((page) => (
              <button key={page} onClick={() => setActivePage(page)} className={`flex w-full items-center justify-between rounded-xl px-3 py-2.5 text-left text-sm font-medium ${activePage === page ? 'bg-brand-50 text-brand-700' : 'text-slate-600 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-800'}`}>
                <span>{page}</span>
                <ArrowRight size={16} />
              </button>
            ))}
          </nav>
          <div className="mt-6 rounded-2xl border border-dashed border-brand-200 bg-brand-50 p-4">
            <div className="flex items-center gap-3">
              <UploadCloud className="text-brand-700" size={18} />
              <div className="text-sm font-medium text-brand-700">Upload data</div>
            </div>
            <button className="mt-4 w-full rounded-xl bg-brand-600 px-3 py-2 text-sm font-semibold text-white">Try sample data</button>
          </div>
        </aside>
        <main className="flex-1 p-4 md:p-8">
          <header className="mb-6 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-2xl font-bold">{activePage}</h1>
              <p className="text-sm text-slate-500">Compute-first insight for cash flow risk.</p>
            </div>
            <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200">
              <ShieldAlert size={16} className="text-emerald-600" />
              Risk status: SAFE
            </div>
          </header>
          {isServerWarming ? (
            <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200">Server warming up…</div>
          ) : (
            pageMap[activePage]
          )}
        </main>
      </div>
    </div>
  );
}
