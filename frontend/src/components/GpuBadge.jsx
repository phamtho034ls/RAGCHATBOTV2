import React, { useEffect, useState } from "react";
import { Cpu } from "lucide-react";
import { getGpuStatus } from "../api/client";

export default function GpuBadge() {
  const [gpu, setGpu] = useState(null);

  useEffect(() => {
    getGpuStatus()
      .then(setGpu)
      .catch(() => setGpu({ available: false }));
  }, []);

  if (!gpu) return null;

  return (
    <div
      className={`flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg ${
        gpu.available
          ? "bg-green-500/10 text-green-400 border border-green-500/20"
          : "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20"
      }`}
    >
      <Cpu size={13} />
      {gpu.available ? (
        <span className="truncate" title={gpu.device_name}>
          GPU: {gpu.device_name?.split(" ").slice(-2).join(" ") || "CUDA"}
        </span>
      ) : (
        <span>CPU only</span>
      )}
    </div>
  );
}
