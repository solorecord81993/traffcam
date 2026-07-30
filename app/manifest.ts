import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "RoadGuard AI",
    short_name: "RoadGuard",
    description:
      "ผู้ช่วยมองวัตถุและเตือนความเสี่ยงระหว่างเดิน ขี่ และขับรถด้วย YOLO",
    start_url: "/",
    display: "standalone",
    background_color: "#070b10",
    theme_color: "#070b10",
    orientation: "any",
    icons: [
      {
        src: "/favicon.svg",
        sizes: "any",
        type: "image/svg+xml",
        purpose: "maskable",
      },
    ],
  };
}
