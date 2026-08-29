import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Reco Trainer · Local Training',
  description: 'Local-only video preparation and RF-DETR training for Reco Cam.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="de"><body>{children}</body></html>;
}
