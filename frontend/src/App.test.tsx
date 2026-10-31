import { render, screen, waitFor } from '@testing-library/react';
import App from './App';

describe('App', () => {
  it('renders the dashboard shell', () => {
    render(<App />);
    expect(screen.getByRole('heading', { name: 'Dashboard' })).toBeInTheDocument();
  });

  it('renders the safe verdict card copy', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText('SAFE')).toBeInTheDocument(), { timeout: 1500 });
  });
});
