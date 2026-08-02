import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RoadGuard AI — ผู้ช่วยเตือนสิ่งกีดขวาง",
  description:
    "เว็บแอปกล้องบน iPhone ที่ตรวจสิ่งกีดขวางแบบเรียลไทม์และเตือนความเสี่ยงระหว่างเดิน ขี่ และขับรถ",
  applicationName: "RoadGuard AI",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "RoadGuard",
  },
  manifest: "/manifest.webmanifest",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: "cover",
  themeColor: "#070b10",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="th">
      <body>{children}</body>
    </html>
  );
}
