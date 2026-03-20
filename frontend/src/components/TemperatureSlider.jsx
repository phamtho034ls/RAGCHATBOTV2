import React from "react";
import { Thermometer } from "lucide-react";

export default function TemperatureSlider({ value, onChange }) {
  const label =
    value <= 0.3 ? "Chính xác" : value <= 0.7 ? "Cân bằng" : "Sáng tạo";

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-gray-400 flex items-center gap-1">
          <Thermometer size={12} /> Độ sáng tạo
        </span>
        <span className="text-gray-300 font-medium">
          {value.toFixed(1)} – {label}
        </span>
      </div>
      <input
        type="range"
        min="0"
        max="1.0"
        step="0.1"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-surface-hover rounded-full appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:w-3.5
          [&::-webkit-slider-thumb]:h-3.5
          [&::-webkit-slider-thumb]:bg-primary-400
          [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:cursor-pointer
          [&::-webkit-slider-thumb]:shadow-md"
      />
    </div>
  );
}
