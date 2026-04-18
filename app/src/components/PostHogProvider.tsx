'use client';

import { useEffect, type ReactNode } from 'react';

const POSTHOG_KEY = process.env.NEXT_PUBLIC_POSTHOG_KEY;
const POSTHOG_HOST =
  process.env.NEXT_PUBLIC_POSTHOG_HOST ?? 'https://us.i.posthog.com';

let initialized = false;

export function PostHogProvider({ children }: { children: ReactNode }) {
  useEffect(() => {
    if (!POSTHOG_KEY || initialized) return;
    initialized = true;
    import('posthog-js').then(({ default: posthog }) => {
      posthog.init(POSTHOG_KEY, {
        api_host: POSTHOG_HOST,
        capture_pageview: true,
        autocapture: true,
        session_recording: {
          maskAllInputs: false,
        },
        persistence: 'localStorage+cookie',
      });
    });
  }, []);

  return <>{children}</>;
}
