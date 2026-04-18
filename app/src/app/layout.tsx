import type { ReactNode } from 'react';
import './globals.css';
import { PostHogProvider } from '../components/PostHogProvider';

export const metadata = {
  title: 'Trademark Risk Check',
  description: 'Pre-filing trademark risk intelligence powered by ML + agentic review.',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <PostHogProvider>{children}</PostHogProvider>
      </body>
    </html>
  );
}
